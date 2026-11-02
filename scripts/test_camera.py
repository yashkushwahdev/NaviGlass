"""Cross-platform Camera Diagnostic Utility for Raspberry Pi 4 & Desktop.

Tests camera initialization (Picamera2 / OpenCV), frame capture, resolution, FPS, and optional preview.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from naviglass.camera import CameraError, OpenCVCamera
from naviglass.config import CameraConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="NaviGlass Camera Diagnostic (Raspberry Pi 4 & Desktop)")
    parser.add_argument("--index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "picamera2", "opencv", "v4l2", "dshow"], help="Camera capture backend")
    parser.add_argument("--source", type=str, default="", help="Video source or camera device ('picamera2' for CSI module)")
    parser.add_argument("--frames", type=int, default=15, help="Number of benchmark frames (default: 15)")
    parser.add_argument("--save", type=str, default="camera_test.jpg", help="Path to save captured frame")
    parser.add_argument("--no-preview", action="store_true", help="Do not open preview GUI window")
    args = parser.parse_args()

    cfg = CameraConfig(
        index=args.index,
        source=args.source,
        backend=args.backend,
        preview=not args.no_preview,
    )
    camera = OpenCVCamera(cfg)

    print("==================================================")
    print("        NaviGlass Camera Diagnostic Utility       ")
    print("==================================================")
    print(f"Device Index:    {cfg.index}")
    print(f"Device Source:   {cfg.source or '(Default Camera)'}")
    print(f"Backend Target:  {cfg.backend}")
    print(f"Preview Window:  {cfg.preview}")
    print("--------------------------------------------------")

    try:
        camera.start()
        print(f"Benchmarking capture of {args.frames} frames...")

        latencies = []
        last_frame = None

        for i in range(args.frames):
            t0 = time.time()
            success, frame = camera.read_frame()
            dt = (time.time() - t0) * 1000.0

            if not success or frame is None:
                print(f"[ERROR] Frame {i+1} capture failed!")
                continue

            latencies.append(dt)
            last_frame = frame

            if cfg.preview:
                should_continue = camera.show_preview(frame, fps=(1000.0 / dt if dt > 0 else 0.0))
                if not should_continue:
                    print("\n[INFO] Preview closed by user ('q' pressed).")
                    break

            time.sleep(0.01)

        if not latencies:
            print("[FAILURE] No frames were captured from camera.", file=sys.stderr)
            sys.exit(1)

        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        print("\nDiagnostic Results:")
        print(f"  Captured Frames: {len(latencies)}/{args.frames}")
        print(f"  Frame Shape:     {last_frame.shape[1]}x{last_frame.shape[0]} (Channels: {last_frame.shape[2]})")
        print(f"  Average Latency: {avg_lat:.2f}ms (Min: {min_lat:.2f}ms, Max: {max_lat:.2f}ms)")
        print(f"  Capture FPS:     {fps:.1f}")

        if args.save and last_frame is not None:
            out_p = Path(args.save)
            cv2.imwrite(str(out_p), last_frame)
            print(f"  Test Frame:      Saved to {out_p.resolve()}")

        print("--------------------------------------------------")
        print("Camera diagnostic PASSED.")

    except CameraError as e:
        print(f"\n[ERROR] Camera Diagnostic FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
