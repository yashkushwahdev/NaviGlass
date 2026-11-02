"""Headless model inference diagnostic on an image file.

Does not require a webcam. Verifies YOLO model loading, inference, and detection parsing.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np
from naviglass.config import ModelConfig
from naviglass.detection import DetectorError, ObjectDetector


def main() -> None:
    default_model = "models/yolov8n.onnx" if Path("models/yolov8n.onnx").exists() else "yolov8n.pt"
    parser = argparse.ArgumentParser(description="NaviGlass Model Inference Diagnostic")
    parser.add_argument("--image", type=str, default="bus.jpg", help="Path to input test image")
    parser.add_argument("--model", type=str, default=default_model, help="Primary model name/path")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "onnx", "ultralytics"], help="Inference backend")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--size", type=int, default=320, help="Inference resolution")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        # Fallback to local project root or current working directory
        project_root_img = Path(__file__).resolve().parent.parent / args.image
        cwd_img = Path.cwd() / args.image
        if project_root_img.exists():
            img_path = project_root_img
        elif cwd_img.exists():
            img_path = cwd_img
        else:
            print(f"[ERROR] Test image '{args.image}' not found.", file=sys.stderr)
            sys.exit(1)

    print("==================================================")
    print("      NaviGlass Desktop Inference Diagnostic      ")
    print("==================================================")
    print(f"Image Source:     {img_path.resolve()}")
    print(f"Model Target:     {args.model}")
    print(f"Inference Size:   {args.size}x{args.size}")
    print(f"Confidence Limit: {args.conf}")
    print("--------------------------------------------------")

    cfg = ModelConfig(
        model_name=args.model,
        backend=args.backend,
        inference_size=args.size,
        confidence_threshold=args.conf,
    )

    detector = ObjectDetector(cfg)
    try:
        detector.load()
    except DetectorError as e:
        print(f"[ERROR] Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load image as BGR numpy array
    image_np = np.array(Image.open(img_path).convert("RGB"))
    image_bgr = image_np[:, :, ::-1]

    print(f"Input Image Size: {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    print("Executing model inference...")

    # Measure cold-start inference
    t_cold = time.perf_counter()
    detections = detector.detect(image_bgr)
    cold_time_ms = (time.perf_counter() - t_cold) * 1000.0

    # Warmup runs
    for _ in range(3):
        detector.detect(image_bgr)

    # Benchmark warm steady-state inference
    num_runs = 20
    latencies = []
    for _ in range(num_runs):
        t_iter = time.perf_counter()
        detector.detect(image_bgr)
        latencies.append((time.perf_counter() - t_iter) * 1000.0)

    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    infer_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print("--------------------------------------------------")
    print(f"Active Model:          {detector.active_model_name}")
    print(f"Cold Start Latency:    {cold_time_ms:.1f}ms (initial compile/load)")
    print(f"Warm Inference Mean:   {mean_ms:.2f}ms (Inference Rate: {infer_fps:.1f} FPS)")
    print(f"Warm Median Latency:   {median_ms:.2f}ms")
    print(f"Warm P95 Latency:      {p95_ms:.2f}ms")
    print(f"Warm Min/Max Latency:  {min_ms:.2f}ms / {max_ms:.2f}ms")
    print(f"Total Detections:      {len(detections)}")
    print("--------------------------------------------------")
    print(f"{'Class':<15} {'Confidence':<12} {'Bounding Box (x1, y1, x2, y2)'}")
    print("-" * 60)
    for d in detections:
        box_str = f"({d.box[0]:.0f}, {d.box[1]:.0f}, {d.box[2]:.0f}, {d.box[3]:.0f})"
        print(f"{d.class_name:<15} {d.confidence:<12.2f} {box_str}")

    detector.close()
    print("==================================================")
    print("Inference diagnostic PASSED.")


if __name__ == "__main__":
    main()
