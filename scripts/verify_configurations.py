"""Verification script for NaviGlass Desktop — 60-Second Stress Test across 4 Configurations.

Tests:
  Config A: Camera only (no YOLO, no depth, no TTS)
  Config B: Camera + YOLO (no depth, no TTS)
  Config C: Camera + YOLO + Depth estimation (no TTS)
  Config D: Camera + YOLO + Depth estimation + TTS (Full pipeline)

Each configuration runs for at least 60 seconds.
Measures and records:
  - Duration (seconds)
  - Camera grab FPS
  - Pipeline processing FPS
  - YOLO inference latency (avg, min, max ms)
  - Depth inference latency (avg, min, max ms)
  - Process RAM usage (RSS MB)
  - CPU usage (% process and system)
  - Total frames captured and processed
  - Zero-freeze verification (no stalls > 1.5s)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil

from audio.speech import AudioConfig, SpeechEngine
from camera.webcam import CameraConfig, OpenCVCamera, Webcam
from config.settings import AlertConfig, AppConfig, DepthConfig, ModelConfig
from depth.depth_estimator import DepthEstimator
from detection.detector import ObjectDetector
from detection.filtering import DetectionFilter
from detection.nearest_object import NearestObjectManager
from utils.logging_config import setup_logging, get_logger

logger = get_logger("verification")


def run_configuration_test(
    name: str,
    duration_sec: float = 60.0,
    enable_yolo: bool = False,
    enable_depth: bool = False,
    enable_tts: bool = False,
    camera_index: int = 0,
) -> Dict[str, Any]:
    """Runs a single pipeline configuration for duration_sec and collects telemetry."""
    print(f"\n=======================================================")
    print(f" Starting Test: {name}")
    print(f" Target Duration: {duration_sec}s")
    print(f" Features: YOLO={enable_yolo}, Depth={enable_depth}, TTS={enable_tts}")
    print(f"=======================================================")

    process = psutil.Process(os.getpid())

    # 1. Configs
    cam_cfg = CameraConfig(index=camera_index, width=640, height=480, fps=30)
    model_cfg = ModelConfig(device="cpu", inference_size=320, confidence_threshold=0.45)
    depth_cfg = DepthConfig(device="cpu", input_size=322, enabled=enable_depth)
    alert_cfg = AlertConfig()
    audio_cfg = AudioConfig(enabled=enable_tts)

    # 2. Subsystems
    speech = None
    if enable_tts:
        speech = SpeechEngine(audio_cfg)
        speech.start()

    detector = None
    if enable_yolo:
        detector = ObjectDetector(model_cfg)
        detector.load()

    depth_estimator = None
    if enable_depth:
        depth_estimator = DepthEstimator(depth_cfg)
        depth_estimator.load()

    filter_mgr = DetectionFilter(alert_cfg)
    nearest_mgr = NearestObjectManager(alert_cfg, depth_cfg)

    # 3. Camera
    camera = Webcam(cam_cfg)
    camera.start()

    # 4. Independent Depth Worker (non-blocking)
    depth_lock = threading.Lock()
    latest_depth = {"map": None, "ms": 0.0}
    depth_busy = threading.Lock()
    stop_depth = threading.Event()
    depth_latencies: List[float] = []

    def depth_worker_loop():
        last_fid = -1
        while not stop_depth.is_set():
            if not enable_depth or depth_estimator is None or not depth_estimator.is_loaded:
                time.sleep(0.05)
                continue
            ok, frame, fid = camera.read_latest_frame()
            if not ok or frame is None or fid == last_fid:
                time.sleep(0.01)
                continue
            acquired = depth_busy.acquire(blocking=False)
            if not acquired:
                time.sleep(0.01)
                continue
            try:
                t0 = time.perf_counter()
                dmap = depth_estimator.estimate_depth_map(frame)
                d_ms = (time.perf_counter() - t0) * 1000.0
                depth_latencies.append(d_ms)
                with depth_lock:
                    latest_depth["map"] = dmap
                    latest_depth["ms"] = d_ms
                last_fid = fid
            except Exception as e:
                logger.error("Depth inference error: %s", e)
            finally:
                depth_busy.release()
            time.sleep(0.005)

    depth_thread = None
    if enable_depth:
        depth_thread = threading.Thread(target=depth_worker_loop, name=f"DepthWorker_{name}", daemon=True)
        depth_thread.start()

    # 5. Pipeline Loop
    yolo_latencies: List[float] = []
    cycle_times: List[float] = []
    ram_samples: List[float] = []
    cpu_samples: List[float] = []
    freeze_events: List[float] = []  # cycle times > 1.5s

    t_start = time.perf_counter()
    last_report_t = t_start
    frame_count = 0
    last_frame_id = -1
    last_detections = []

    try:
        while True:
            t_cycle_start = time.perf_counter()
            elapsed_total = t_cycle_start - t_start
            if elapsed_total >= duration_sec:
                break

            # Read frame from camera
            ok, frame, fid = camera.read_latest_frame()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1

            # YOLO inference
            infer_ms = 0.0
            if enable_yolo and detector is not None:
                if fid != last_frame_id:
                    t_y = time.perf_counter()
                    try:
                        last_detections = detector.detect(frame)
                    except Exception as e:
                        logger.error("YOLO error: %s", e)
                    infer_ms = (time.perf_counter() - t_y) * 1000.0
                    yolo_latencies.append(infer_ms)
                    last_frame_id = fid

            # Depth association
            depth_ms = 0.0
            if enable_depth and depth_estimator is not None:
                with depth_lock:
                    dmap = latest_depth["map"]
                    depth_ms = latest_depth["ms"]
                dets_with_dist = depth_estimator.attach_distances_to_detections(
                    last_detections, dmap, current_time=time.time()
                )
            else:
                dets_with_dist = last_detections

            # Nearest object and alerts
            if enable_yolo:
                pos_dets = [filter_mgr.attach_position_metadata(d, frame.shape[1]) for d in dets_with_dist]
                enriched, nearest = nearest_mgr.process_frame(pos_dets, current_time=time.time(), frame_width=frame.shape[1])

                if enable_tts and speech is not None and nearest is not None:
                    alert = nearest_mgr.evaluate_voice_alert(nearest, current_time=time.time())
                    if alert:
                        phrase, _, urgent = alert
                        speech.speak_async(phrase, priority=urgent)

            t_cycle_end = time.perf_counter()
            dt = t_cycle_end - t_cycle_start
            cycle_times.append(dt)
            if dt > 1.5:
                freeze_events.append(dt)
                logger.warning("Freeze/stall detected: %.2fs", dt)

            # Periodic diagnostics report every 1.0 second (Requirement 2)
            if t_cycle_end - last_report_t >= 1.0:
                cur_wall = time.time()
                cur_ram = process.memory_info().rss / (1024 * 1024)
                cur_cpu = process.cpu_percent()
                ram_samples.append(cur_ram)
                cpu_samples.append(cur_cpu)
                recent_fps = 1.0 / dt if dt > 0 else 0.0
                cam_fps = camera.grab_fps
                cam_alive = camera.grabber_alive
                succ_reads = camera.successful_reads
                fail_reads = camera.failed_reads
                cap_ts = camera.latest_capture_timestamp
                disp_ts = cur_wall
                depth_busy_flag = depth_busy.locked()

                cap_stall_sec = cur_wall - cap_ts if cap_ts > 0 else 0.0
                if cap_ts > 0 and cap_stall_sec > 1.5:
                    logger.warning("[STALL DETECTED] Camera capture timestamp stopped advancing! Elapsed: %.2fs", cap_stall_sec)

                # Log diagnostics every second
                logger.info(
                    "[DIAGNOSTICS 1s] CamAlive=%s | SuccReads=%d | FailReads=%d | LatestCapTS=%.3f | LatestDispTS=%.3f | CamFPS=%.1f | LoopFPS=%.1f | YOLO=%.1fms | Depth=%.1fms | DepthBusy=%s | RAM=%.1fMB",
                    cam_alive, succ_reads, fail_reads, cap_ts, disp_ts, cam_fps, recent_fps, infer_ms, depth_ms, depth_busy_flag, cur_ram,
                )
                if int(elapsed_total) % 10 == 0:
                    print(
                        f"  [{elapsed_total:4.1f}s / {duration_sec}s] "
                        f"CamGrabFPS={cam_fps:4.1f} | LoopFPS={recent_fps:4.1f} | "
                        f"YOLO={infer_ms:4.1f}ms | Depth={depth_ms:4.1f}ms | "
                        f"RAM={cur_ram:5.1f}MB | CPU={cur_cpu:4.1f}%"
                    )
                last_report_t = t_cycle_end

            # Small yield
            time.sleep(0.005)

    finally:
        stop_depth.set()
        if depth_thread and depth_thread.is_alive():
            depth_thread.join(timeout=2.0)
        camera.stop()
        if detector:
            detector.close()
        if depth_estimator:
            depth_estimator.close()
        if speech:
            speech.stop()

    actual_duration = time.perf_counter() - t_start
    avg_loop_fps = frame_count / actual_duration if actual_duration > 0 else 0.0
    cam_grab_fps = camera.grab_fps

    res = {
        "name": name,
        "completed_60s": actual_duration >= (duration_sec - 1.0),
        "actual_duration_sec": round(actual_duration, 2),
        "total_frames": frame_count,
        "camera_grab_fps": round(cam_grab_fps, 2),
        "loop_fps": round(avg_loop_fps, 2),
        "yolo_latency_avg_ms": round(float(np.mean(yolo_latencies)), 1) if yolo_latencies else 0.0,
        "yolo_latency_min_ms": round(float(np.min(yolo_latencies)), 1) if yolo_latencies else 0.0,
        "yolo_latency_max_ms": round(float(np.max(yolo_latencies)), 1) if yolo_latencies else 0.0,
        "depth_latency_avg_ms": round(float(np.mean(depth_latencies)), 1) if depth_latencies else 0.0,
        "depth_latency_min_ms": round(float(np.min(depth_latencies)), 1) if depth_latencies else 0.0,
        "depth_latency_max_ms": round(float(np.max(depth_latencies)), 1) if depth_latencies else 0.0,
        "depth_inferences_total": len(depth_latencies),
        "ram_mb_peak": round(max(ram_samples) if ram_samples else process.memory_info().rss / (1024 * 1024), 1),
        "ram_mb_final": round(process.memory_info().rss / (1024 * 1024), 1),
        "cpu_percent_avg": round(float(np.mean(cpu_samples)), 1) if cpu_samples else 0.0,
        "freeze_count": len(freeze_events),
        "status": "PASSED (NO FREEZE)" if len(freeze_events) == 0 and actual_duration >= (duration_sec - 1.0) else "FAILED",
    }

    print(f"\n--- {name} Results ---")
    print(f"  Duration:           {res['actual_duration_sec']}s (Goal: {duration_sec}s) -> {'PASS' if res['completed_60s'] else 'FAIL'}")
    print(f"  Total Frames:       {res['total_frames']}")
    print(f"  Camera Grab FPS:    {res['camera_grab_fps']}")
    print(f"  Loop/Preview FPS:   {res['loop_fps']}")
    if enable_yolo:
        print(f"  YOLO Latency:       {res['yolo_latency_avg_ms']}ms avg ({res['yolo_latency_min_ms']}-{res['yolo_latency_max_ms']}ms)")
    if enable_depth:
        print(f"  Depth Latency:      {res['depth_latency_avg_ms']}ms avg ({res['depth_latency_min_ms']}-{res['depth_latency_max_ms']}ms) [{res['depth_inferences_total']} inferences]")
    print(f"  RAM Peak:           {res['ram_mb_peak']} MB")
    print(f"  Freezes (>1.5s):    {res['freeze_count']}")
    print(f"  Overall Status:     {res['status']}")

    return res


def main():
    setup_logging()
    print("=================================================================")
    print("  NaviGlass Desktop — Verification Benchmark (4 x 60 Seconds)    ")
    print("=================================================================")

    target_duration = 60.0

    # Test A: Camera only
    res_a = run_configuration_test(
        name="Configuration A: Camera Only",
        duration_sec=target_duration,
        enable_yolo=False,
        enable_depth=False,
        enable_tts=False,
    )

    # Test B: Camera + YOLO
    res_b = run_configuration_test(
        name="Configuration B: Camera + YOLO",
        duration_sec=target_duration,
        enable_yolo=True,
        enable_depth=False,
        enable_tts=False,
    )

    # Test C: Camera + YOLO + Depth
    res_c = run_configuration_test(
        name="Configuration C: Camera + YOLO + Depth",
        duration_sec=target_duration,
        enable_yolo=True,
        enable_depth=True,
        enable_tts=False,
    )

    # Test D: Camera + YOLO + Depth + TTS
    res_d = run_configuration_test(
        name="Configuration D: Camera + YOLO + Depth + TTS",
        duration_sec=target_duration,
        enable_yolo=True,
        enable_depth=True,
        enable_tts=True,
    )

    print("\n" + "=" * 65)
    print("           FINAL VERIFICATION SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Config':<35} | {'60s?':<5} | {'Cam FPS':<8} | {'Loop FPS':<9} | {'YOLO ms':<8} | {'Depth ms':<9} | {'Freezes':<7} | {'Status'}")
    print("-" * 105)
    for r in [res_a, res_b, res_c, res_d]:
        y_ms = f"{r['yolo_latency_avg_ms']:.1f}" if r['yolo_latency_avg_ms'] > 0 else "—"
        d_ms = f"{r['depth_latency_avg_ms']:.1f}" if r['depth_latency_avg_ms'] > 0 else "—"
        ok_60 = "YES" if r['completed_60s'] else "NO"
        print(f"{r['name']:<35} | {ok_60:<5} | {r['camera_grab_fps']:<8.1f} | {r['loop_fps']:<9.1f} | {y_ms:<8} | {d_ms:<9} | {r['freeze_count']:<7} | {r['status']}")
    print("=" * 105)


if __name__ == "__main__":
    main()

