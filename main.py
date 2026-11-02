"""NaviGlass Desktop — AI-Powered Assistive Vision System.

Main application entry point supporting both the accessible Desktop GUI (default)
and headless / terminal preview modes.

Architecture:
- Camera grabber thread continuously drains the OS buffer (inside Webcam).
- YOLO detection runs in the main CLI loop or a dedicated worker thread.
- Depth inference runs in a separate background thread (never blocking camera/GUI).
- TTS runs in its own worker thread (inside SpeechEngine).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

from audio.speech import SpeechEngine
from camera.webcam import CameraError, OpenCVCamera
from config.settings import AppConfig, ConfigurationError
from depth.depth_estimator import DepthEstimator, DepthEstimatorError, MetricDepthMap
from detection.detector import DetectorError, ObjectDetector
from detection.filtering import DetectionFilter
from detection.nearest_object import NearestObjectManager, WarningDistanceLevel
from utils.logging_config import get_logger, setup_logging

logger = get_logger("main")


def _get_ram_mb() -> float:
    """Returns current process RSS in MB, or -1 if psutil unavailable."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    default_cfg = "config.rpi4.yaml" if (sys.platform.startswith("linux") and Path("config.rpi4.yaml").exists()) else "config.yaml"
    parser = argparse.ArgumentParser(
        description="NaviGlass: AI-Powered Assistive Vision System (Raspberry Pi 4 & Desktop)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", default=True, help="Launch Desktop GUI application (default)")
    parser.add_argument("--cli", action="store_true", help="Launch in CLI/headless console mode instead of GUI")
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to config yaml file")
    parser.add_argument("--source", type=str, default=None, help="Video file or image path (overrides webcam)")
    parser.add_argument("--camera-index", type=int, default=None, help="Webcam device index (default: 0)")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold (0.0-1.0)")
    parser.add_argument("--cooldown", type=float, default=None, help="Override alert cooldown seconds")
    parser.add_argument("--interval", type=float, default=None, help="Override announcement interval in seconds (default: 3.0)")
    parser.add_argument("--no-depth", action="store_true", help="Disable Depth Anything V2 metric depth estimation")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window in CLI mode")
    parser.add_argument("--no-audio", action="store_true", help="Disable spoken audio alerts")
    parser.add_argument("--max-frames", type=int, default=None, help="Exit after processing N frames (for benchmarks)")
    return parser.parse_args()


def run_cli_mode(config: AppConfig, args: argparse.Namespace) -> None:
    """Runs console processing loop with optional OpenCV video preview window.

    Depth inference runs in a dedicated background thread so that
    the camera capture and YOLO detection loop never stalls.
    """
    use_depth = config.depth.enabled and not getattr(args, "no_depth", False)

    print("=========================================================")
    print("      NaviGlass Desktop: Assistive Smart Glasses         ")
    print("=========================================================")
    print(f"Webcam Index:         {config.camera.index} (Source: {config.camera.source or 'Live Webcam'})")
    print(f"Model:                {config.model.model_name}")
    print(f"Confidence Limit:     {config.model.confidence_threshold}")
    print(f"Inference Size:       {config.model.inference_size}x{config.model.inference_size}")
    print(f"Metric Depth:         {use_depth} (Model: {config.depth.model_name if use_depth else 'Off'})")
    print(f"Alert Cooldown:       {config.alert.cooldown_seconds}s")
    print(f"Voice Alerts:         {config.audio.enabled}")
    print("---------------------------------------------------------")

    speech = SpeechEngine(config.audio)
    speech.start()

    detector = ObjectDetector(config.model)
    try:
        detector.load()
    except DetectorError as e:
        logger.fatal("Model loading failed: %s", e)
        speech.stop()
        sys.exit(1)

    depth_estimator = DepthEstimator(config.depth)
    depth_loaded = False
    if use_depth:
        try:
            depth_estimator.load()
            depth_loaded = True
        except DepthEstimatorError as e:
            logger.warning("Depth estimator failed to load: %s. Continuing in 2D mode.", e)

    camera = OpenCVCamera(config.camera)
    try:
        camera.start()
    except CameraError as e:
        logger.fatal("Camera startup failed: %s", e)
        speech.stop()
        detector.close()
        depth_estimator.close()
        sys.exit(1)

    filter_mgr = DetectionFilter(config.alert, min_confidence=config.model.confidence_threshold)
    nearest_mgr = NearestObjectManager(config.alert, config.depth)

    # ── Depth worker thread (non-blocking) ──────────────────────────────
    depth_lock = threading.Lock()
    latest_depth_map: dict = {"map": None, "ms": 0.0}
    depth_busy = threading.Lock()
    stop_depth = threading.Event()

    def _depth_worker() -> None:
        """Continuously runs depth inference on the latest camera frame."""
        last_fid = -1
        while not stop_depth.is_set():
            if not depth_loaded:
                time.sleep(0.1)
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
                ms = (time.perf_counter() - t0) * 1000.0
                with depth_lock:
                    latest_depth_map["map"] = dmap
                    latest_depth_map["ms"] = ms
                last_fid = fid
            except Exception as e:
                logger.error("Depth worker exception: %s", e)
            finally:
                depth_busy.release()
            time.sleep(0.005)

    depth_thread = None
    if depth_loaded:
        depth_thread = threading.Thread(target=_depth_worker, name="CLIDepthWorker", daemon=True)
        depth_thread.start()
        logger.info("CLI depth worker thread started.")
    # ────────────────────────────────────────────────────────────────────

    # Initial announcement
    speech.speak_async("NaviGlass Desktop ready.")
    print("\n[ACTIVE] Processing webcam feed. Press 'q' or Ctrl+C to stop.\n")

    fps_tracker = deque(maxlen=config.monitoring.fps_window_size)
    frame_idx = 0
    last_detections = []
    window_name = "NaviGlass Desktop Preview"
    log_interval_t = time.perf_counter()

    try:
        while True:
            t_start = time.perf_counter()

            # 1. Capture latest frame (from continuous grabber thread — non-blocking)
            success, frame = camera.read_frame()
            if not success or frame is None:
                if config.camera.source:
                    print("\n[INFO] End of media source reached.")
                    break
                time.sleep(0.01)
                continue

            frame_idx += 1

            # 2. Detect Objects (every N frames)
            infer_ms = 0.0
            if frame_idx % config.model.frame_skip == 0:
                t_inf = time.perf_counter()
                try:
                    last_detections = detector.detect(frame, timestamp=time.time())
                    infer_ms = (time.perf_counter() - t_inf) * 1000.0
                except Exception as e:
                    logger.error("YOLO detection exception: %s", e)

            # 3. Read latest depth result (non-blocking, from depth worker)
            depth_ms = 0.0
            depth_status = ""
            if depth_loaded:
                with depth_lock:
                    current_depth = latest_depth_map["map"]
                    depth_ms = latest_depth_map["ms"]
                depth_status = "Metric VITS"
                detections_with_dist = depth_estimator.attach_distances_to_detections(
                    last_detections,
                    current_depth,
                    current_time=time.time(),
                )
            else:
                depth_status = "Disabled" if not use_depth else "Offline"
                detections_with_dist = last_detections

            # 4. Attach Position Guidance Metadata
            if frame is not None and detections_with_dist:
                positioned_dets = [
                    filter_mgr.attach_position_metadata(d, frame_width=frame.shape[1])
                    for d in detections_with_dist
                ]
            else:
                positioned_dets = []

            # 5. Closest Object Prioritization & Temporal Distance Smoothing
            enriched_dets, nearest_obj = nearest_mgr.process_frame(
                positioned_dets,
                current_time=time.time(),
                frame_width=frame.shape[1] if frame is not None else 640,
            )

            # 6. Spoken Alert Decision Logic
            if depth_loaded and nearest_obj is not None:
                voice_alert = nearest_mgr.evaluate_voice_alert(nearest_obj, current_time=time.time())
                if voice_alert:
                    phrase, det_obj, is_urgent = voice_alert
                    dist_lbl = f"{det_obj.distance:.1f}m" if det_obj.distance is not None else "N/A"
                    print(f'\n >>> [VOICE CLOSEST{" !" if is_urgent else ""}]: "{phrase}" (dist: {dist_lbl}, pos: {det_obj.position})')
                    speech.speak_async(phrase, priority=is_urgent)
            else:
                alert_result = filter_mgr.evaluate_alert(
                    enriched_dets or positioned_dets,
                    current_time=time.time(),
                    confidence_threshold=config.model.confidence_threshold,
                    frame_width=frame.shape[1] if frame is not None else 640,
                )
                if alert_result:
                    phrase, det_obj = alert_result
                    print(f'\n >>> Spoken: "{phrase}" (conf: {det_obj.confidence:.2f})')
                    speech.speak_async(phrase)

            # 7. Measure Processing Rate
            duration = time.perf_counter() - t_start
            if duration > 0:
                fps_tracker.append(1.0 / duration)
            e2e_fps = sum(fps_tracker) / len(fps_tracker) if fps_tracker else 0.0

            # 8. Terminal Telemetry
            det_strs = []
            for d in (enriched_dets or positioned_dets):
                tag = "[C]" if d.is_closest else ""
                d_str = f"~{d.distance:.1f}m" if d.distance is not None else "dist:n/a"
                det_strs.append(f"{tag}{d.class_name}({d.confidence:.2f},{d.position or 'C'},{d_str})")
            summary = ", ".join(det_strs) if det_strs else "None"
            print(f"\r[Frame {frame_idx:04d} | E2E: {e2e_fps:4.1f} FPS | YOLO: {infer_ms:4.1f}ms | Depth: {depth_ms:4.1f}ms] {summary:<55}", end="", flush=True)

            # Periodic diagnostic log every 1.0 second (Requirement 2)
            now_t = time.perf_counter()
            if now_t - log_interval_t >= 1.0:
                cur_wall_time = time.time()
                cap_ts = camera.latest_capture_timestamp
                disp_ts = cur_wall_time
                cam_alive = camera.grabber_alive
                succ_reads = camera.successful_reads
                fail_reads = camera.failed_reads
                depth_busy_flag = depth_busy.locked()
                ram = _get_ram_mb()
                ram_str = f"{ram:.0f}MB" if ram >= 0 else "N/A"
                grab_fps = camera.grab_fps

                cap_stall_sec = cur_wall_time - cap_ts if cap_ts > 0 else 0.0
                if cap_ts > 0 and cap_stall_sec > 1.5:
                    logger.warning(
                        "[STALL DETECTED] Camera capture timestamp stopped advancing! Elapsed: %.2fs",
                        cap_stall_sec,
                    )

                logger.info(
                    "[DIAGNOSTICS 1s] CamAlive=%s | SuccReads=%d | FailReads=%d | LatestCapTS=%.3f | LatestDispTS=%.3f | CamFPS=%.1f | E2E_FPS=%.1f | YOLO=%.1fms | Depth=%.1fms | DepthBusy=%s | RAM=%s",
                    cam_alive, succ_reads, fail_reads, cap_ts, disp_ts, grab_fps, e2e_fps, infer_ms, depth_ms, depth_busy_flag, ram_str,
                )
                log_interval_t = now_t

            # 9. Optional OpenCV Preview Window
            if config.camera.preview and not args.no_preview:
                import cv2
                try:
                    annotated = detector.draw_overlays(
                        frame,
                        enriched_dets or positioned_dets,
                        fps=e2e_fps,
                        infer_ms=infer_ms,
                        depth_ms=depth_ms,
                        depth_status=depth_status,
                    )
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("\n[INFO] Preview window closed by user ('q' pressed).")
                        break
                except cv2.error as e:
                    logger.debug("OpenCV preview disabled (display unavailable): %s", e)
                    config.camera.preview = False

            if args.max_frames and frame_idx >= args.max_frames:
                print(f"\n[INFO] Reached requested limit of {args.max_frames} frames.")
                break

    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Ctrl+C received. Stopping NaviGlass Desktop...")
    finally:
        stop_depth.set()
        if depth_thread and depth_thread.is_alive():
            depth_thread.join(timeout=2.0)
        camera.stop()
        detector.close()
        depth_estimator.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        speech.speak_async("NaviGlass stopped.")
        speech.wait_until_done(timeout=2.0)
        time.sleep(0.3)
        speech.stop()
        print("[SHUTDOWN] All resources released cleanly.")



def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    setup_logging()

    try:
        config = AppConfig.load(args.config)
    except ConfigurationError as e:
        print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # CLI Overrides
    if args.source is not None:
        config.camera.source = args.source
    if args.camera_index is not None:
        config.camera.index = args.camera_index
    if args.conf is not None:
        config.model.confidence_threshold = args.conf
    if args.cooldown is not None:
        config.alert.cooldown_seconds = args.cooldown
        config.alert.announcement_interval_seconds = args.cooldown
    if args.interval is not None:
        config.alert.announcement_interval_seconds = args.interval
        config.alert.cooldown_seconds = args.interval
    if args.no_audio:
        config.audio.enabled = False

    # Choose between GUI mode (default) and CLI mode
    if args.cli:
        run_cli_mode(config, args)
    else:
        try:
            from app.gui import launch_gui
            launch_gui(config)
        except Exception as e:
            err_msg = str(e).lower()
            if "display" in err_msg or "no display name" in err_msg or "tcl" in err_msg:
                logger.warning("GUI launch failed due to missing graphical display (%s). Falling back to CLI mode...", e)
                print("\n[NOTE] Graphical display not detected (headless / SSH environment). Falling back to CLI mode.\n")
                run_cli_mode(config, args)
            else:
                raise


if __name__ == "__main__":
    main()
