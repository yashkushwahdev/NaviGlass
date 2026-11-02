"""NaviGlass Desktop Main Application Orchestrator."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from naviglass.audio import WindowsTTS
from naviglass.camera import CameraError, OpenCVCamera
from naviglass.config import AppConfig, ConfigurationError
from naviglass.detection import DetectorError, ObjectDetector
from naviglass.safety import AlertManager


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NaviGlass Desktop: Windows Prototyping Platform")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", type=str, default=None, help="Video file or image path (overrides webcam)")
    parser.add_argument("--camera-index", type=int, default=None, help="Webcam device index (default 0)")
    parser.add_argument("--no-preview", action="store_true", help="Disable desktop preview GUI window")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold (0.0-1.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Exit after processing N frames")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

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
    if args.no_preview:
        config.camera.preview = False
    if args.conf is not None:
        config.model.confidence_threshold = args.conf

    print("=========================================================")
    print("      NaviGlass Desktop: Assistive Smart Glasses         ")
    print("=========================================================")
    print(f"Webcam Index:         {config.camera.index} (Source: {config.camera.source or 'Live Webcam'})")
    print(f"Model:                {config.model.model_name}")
    print(f"Confidence Limit:     {config.model.confidence_threshold}")
    print(f"Inference Size:       {config.model.inference_size}x{config.model.inference_size}")
    print(f"GUI Preview Window:   {config.camera.preview}")
    print(f"Alert Cooldown:       {config.alert.cooldown_seconds}s")
    print("---------------------------------------------------------")

    # 1. Initialize Subsystems
    tts = WindowsTTS(config.audio)
    tts.start()

    detector = ObjectDetector(config.model)
    try:
        detector.load()
    except DetectorError as e:
        print(f"[FATAL] Model loading failed: {e}", file=sys.stderr)
        tts.stop()
        sys.exit(1)

    camera = OpenCVCamera(config.camera)
    try:
        camera.start()
    except CameraError as e:
        print(f"[FATAL] Camera startup failed: {e}", file=sys.stderr)
        tts.stop()
        detector.close()
        sys.exit(1)

    alert_mgr = AlertManager(config.alert)

    # Startup chime / announcement
    tts.speak_async("NaviGlass Desktop ready.")

    print("\n[ACTIVE] Processing webcam feed. Press 'q' in preview window or Ctrl+C to stop.\n")

    fps_tracker = deque(maxlen=config.monitoring.fps_window_size)
    frame_idx = 0
    last_detections = []

    try:
        while True:
            t_start = time.time()

            # 2. Capture Frame
            success, frame = camera.read_frame()
            if not success or frame is None:
                if config.camera.source:
                    print("\n[INFO] End of media source reached.")
                    break
                print("\n[WARNING] Failed to capture frame from webcam. Retrying...", end="")
                time.sleep(0.05)
                continue

            frame_idx += 1

            # 3. Detect Objects (every N frames according to frame_skip)
            infer_ms = 0.0
            if frame_idx % config.model.frame_skip == 0:
                t_infer_0 = time.perf_counter()
                last_detections = detector.detect(frame, timestamp=t_start)
                infer_ms = (time.perf_counter() - t_infer_0) * 1000.0

            # 4. Basic Alert Logic
            announcement = alert_mgr.evaluate_detections(last_detections, current_time=t_start)
            if announcement:
                print(f'\n >>> Spoken: "{announcement}"')
                tts.speak_async(announcement)

            # 5. Measure End-to-End Processing Rate
            duration = time.time() - t_start
            if duration > 0:
                fps_tracker.append(1.0 / duration)
            e2e_fps = sum(fps_tracker) / len(fps_tracker) if fps_tracker else 0.0

            # 6. Terminal Output
            det_strs = [f"{d.class_name} ({d.confidence:.2f})" for d in last_detections]
            summary = ", ".join(det_strs) if det_strs else "None"
            print(f"\r[Frame {frame_idx:04d} | E2E: {e2e_fps:4.1f} FPS | Infer: {infer_ms:4.1f}ms] Detected: {summary:<35}", end="", flush=True)

            # 7. Optional GUI Preview
            continue_running = camera.show_preview(frame, last_detections, fps=e2e_fps)
            if not continue_running:
                print("\n[INFO] Preview window closed by user ('q' pressed).")
                break

            if args.max_frames and frame_idx >= args.max_frames:
                print(f"\n[INFO] Reached requested limit of {args.max_frames} frames.")
                break

    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Ctrl+C received. Stopping NaviGlass Desktop...")
    finally:
        try:
            camera.stop()
        except Exception as e:
            print(f"[SHUTDOWN WARNING] Camera cleanup error: {e}")

        try:
            detector.close()
        except Exception as e:
            print(f"[SHUTDOWN WARNING] Detector cleanup error: {e}")

        try:
            tts.speak_async("NaviGlass stopped.")
            tts.wait_until_done(timeout=2.0)
            time.sleep(0.4)
            tts.stop()
        except Exception as e:
            print(f"[SHUTDOWN WARNING] Audio cleanup error: {e}")

        print("[SHUTDOWN] All resources released cleanly.")


if __name__ == "__main__":
    main()
