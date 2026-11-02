"""Benchmark and latency profiler for NaviGlass YOLO + Metric Depth pipeline."""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from config.settings import AppConfig
from detection.detector import ObjectDetector
from depth.depth_estimator import DepthEstimator
from detection.nearest_object import NearestObjectManager

def run_benchmark():
    print("=========================================================")
    print("      NaviGlass Metric Depth & Detection Profiler        ")
    print("=========================================================")

    cfg_path = repo_root / "config.yaml"
    cfg = AppConfig.load(str(cfg_path))
    img_path = repo_root / "bus.jpg"
    if not img_path.exists():
        print(f"Sample image '{img_path}' not found, generating synthetic test image...")
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        frame = cv2.imread(str(img_path))

    print(f"Test Frame: {frame.shape[1]}x{frame.shape[0]}")
    print(f"YOLO Model: {cfg.model.model_name} (Size: {cfg.model.inference_size})")
    print(f"Depth Model: {cfg.depth.model_name} (Input Size: {cfg.depth.input_size})")
    print("---------------------------------------------------------")

    detector = ObjectDetector(cfg.model)
    detector.load()

    depth_estimator = DepthEstimator(cfg.depth)
    depth_estimator.load()

    nearest_mgr = NearestObjectManager(cfg.alert, cfg.depth)

    # Warmup
    print("Warming up models...")
    for _ in range(2):
        dets = detector.detect(frame)
        dmap = depth_estimator.estimate_depth_map(frame)

    print("Running timed inference benchmark (5 iterations)...")
    yolo_latencies = []
    depth_latencies = []
    total_latencies = []

    for i in range(5):
        t0 = time.perf_counter()

        # 1. YOLO
        t_yolo = time.perf_counter()
        detections = detector.detect(frame, timestamp=time.time())
        yolo_ms = (time.perf_counter() - t_yolo) * 1000.0
        yolo_latencies.append(yolo_ms)

        # 2. Depth
        t_dep = time.perf_counter()
        depth_info = depth_estimator.estimate_depth_map(frame)
        depth_ms = (time.perf_counter() - t_dep) * 1000.0
        depth_latencies.append(depth_ms)

        # 3. Association and Nearest Object
        dets_with_dist = depth_estimator.attach_distances_to_detections(detections, depth_info, current_time=time.time())
        enriched, nearest = nearest_mgr.process_frame(dets_with_dist, current_time=time.time(), frame_width=frame.shape[1])

        tot_ms = (time.perf_counter() - t0) * 1000.0
        total_latencies.append(tot_ms)

        print(f" Iteration {i+1}: YOLO = {yolo_ms:.1f}ms | Depth = {depth_ms:.1f}ms | Total = {tot_ms:.1f}ms")

    avg_yolo = float(np.mean(yolo_latencies))
    avg_depth = float(np.mean(depth_latencies))
    avg_total = float(np.mean(total_latencies))

    print("\n=== Benchmark Results ===")
    print(f"Average YOLO26n CPU Latency:       {avg_yolo:.1f} ms (~{1000.0/max(avg_yolo, 0.001):.1f} FPS standalone)")
    print(f"Average Depth Anything V2 Latency: {avg_depth:.1f} ms (~{1000.0/max(avg_depth, 0.001):.1f} FPS standalone)")
    print(f"Average Joint Pipeline Latency:    {avg_total:.1f} ms")
    if nearest:
        voice_str = nearest_mgr.format_nearest_voice_alert(nearest)
        print(f"\nNearest Detected Object: '{nearest.class_name}' at ~{nearest.distance}m ({nearest.position})")
        print(f"Spoken Announcement: '{voice_str}'")

    detector.close()
    depth_estimator.close()
    print("=========================================================")

if __name__ == "__main__":
    run_benchmark()
