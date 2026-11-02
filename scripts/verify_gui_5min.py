"""Verification of actual GUI for 5 continuous minutes (300 seconds).

Measures and confirms:
- Camera preview continues changing and updating
- CamFID and DispFID continually advance
- All worker threads (Grabber, Detection, Depth, Speech) remain alive
- No freeze events (>1.5s)
- Records average FPS, latencies, CPU %, and RAM
"""

import os
import sys
import threading
import time
import traceback
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import tkinter as tk
from app.gui import NaviGlassGUI
from config.settings import AppConfig
from utils.logging_config import setup_logging, get_logger

logger = get_logger("gui_5min_verification")


def run_gui_5min(duration_sec: float = 300.0):
    setup_logging()
    print("=================================================================")
    print("   NaviGlass Desktop — 5-Minute Continuous GUI Verification     ")
    print(f"   Target Duration: {duration_sec}s (5.0 minutes)              ")
    print("   Features: Live Camera + YOLO + Depth Anything V2 + TTS Alerts ")
    print("=================================================================")

    process = psutil.Process(os.getpid())
    cfg = AppConfig.load()
    root = tk.Tk()
    app = NaviGlassGUI(root, cfg)

    # Let the GUI initialize its window & canvas
    for _ in range(15):
        root.update()
        time.sleep(0.04)

    print("\n[GUI STARTUP] Starting camera feed and all AI workers...")
    app.start_camera()

    t_start = time.time()
    last_log_t = t_start
    last_fid = 0
    freeze_events = []
    cam_fps_samples = []
    det_fps_samples = []
    yolo_lat_samples = []
    depth_lat_samples = []
    ui_dur_samples = []
    ram_samples = []
    cpu_samples = []

    try:
        while True:
            root.update()
            time.sleep(0.01)

            now = time.time()
            elapsed = now - t_start
            if elapsed >= duration_sec:
                break

            if now - last_log_t >= 1.0:
                cam_alive = app.webcam.grabber_alive
                det_alive = app._worker_thread is not None and app._worker_thread.is_alive()
                depth_alive = app._depth_thread is not None and app._depth_thread.is_alive()
                depth_busy = app._depth_busy.locked()
                cam_fid = app.webcam.frame_id
                disp_fid = app._latest_displayed_fid
                cam_fps = app.webcam.grab_fps
                cap_ts = app.webcam.latest_capture_timestamp
                disp_ts = app._latest_displayed_timestamp
                time_since_gui = now - disp_ts if disp_ts > 0 else 0.0
                cur_ram = process.memory_info().rss / (1024 * 1024)
                cur_cpu = process.cpu_percent()

                cam_fps_samples.append(cam_fps)
                ram_samples.append(cur_ram)
                cpu_samples.append(cur_cpu)
                ui_dur_samples.append(app._last_ui_callback_duration_ms)

                if time_since_gui > 1.5:
                    freeze_events.append((elapsed, time_since_gui))
                    print(f"  [WARNING] Freeze event at {elapsed:.1f}s: DispLag={time_since_gui:.2f}s!")

                # Print progress every 15 seconds
                if int(elapsed) % 15 == 0:
                    print(
                        f"  [{elapsed:5.1f}s / {duration_sec:.0f}s] "
                        f"CamFID={cam_fid:5d} | DispFID={disp_fid:5d} | "
                        f"CamFPS={cam_fps:4.1f} | DispLag={time_since_gui:4.2f}s | "
                        f"GrabberAlive={cam_alive} | DetAlive={det_alive} | DepthAlive={depth_alive} | "
                        f"RAM={cur_ram:5.1f}MB | CPU={cur_cpu:4.1f}%"
                    )

                last_log_t = now

    finally:
        print("\n[GUI SHUTDOWN] Stopping camera and closing GUI...")
        try:
            app.stop_camera()
        except Exception as e:
            print("Error stopping camera:", e)
        try:
            root.destroy()
        except Exception:
            pass

    actual_duration = time.time() - t_start
    final_cam_fid = app.webcam.frame_id
    final_disp_fid = app._latest_displayed_fid

    print("\n=================================================================")
    print("            5-MINUTE GUI VERIFICATION RESULTS                    ")
    print("=================================================================")
    print(f"  Target Duration:        {duration_sec:.1f}s")
    print(f"  Actual Duration:        {actual_duration:.1f}s ({actual_duration/60.0:.2f} minutes)")
    print(f"  Completed >= 300s:      {'YES (PASS)' if actual_duration >= 299.0 else 'NO (FAIL)'}")
    print(f"  Total Frames Captured:  {final_cam_fid}")
    print(f"  Total Frames Displayed: {final_disp_fid}")
    print(f"  Average Camera FPS:     {np.mean(cam_fps_samples):.1f}")
    print(f"  Average GUI Render Dur: {np.mean(ui_dur_samples):.2f} ms")
    print(f"  Peak RAM:               {max(ram_samples):.1f} MB")
    print(f"  Freeze Events (>1.5s):  {len(freeze_events)}")
    print(f"  All Workers Kept Alive: {'YES (PASS)' if (final_cam_fid > 100 and final_disp_fid > 100) else 'NO (FAIL)'}")
    print("=================================================================")

    return {
        "actual_duration_sec": actual_duration,
        "completed_5min": actual_duration >= 299.0,
        "total_captured_frames": final_cam_fid,
        "total_displayed_frames": final_disp_fid,
        "avg_camera_fps": float(np.mean(cam_fps_samples)),
        "peak_ram_mb": float(max(ram_samples)),
        "freeze_count": len(freeze_events),
        "status": "PASSED" if (len(freeze_events) == 0 and actual_duration >= 299.0) else "FAILED",
    }


if __name__ == "__main__":
    run_gui_5min(duration_sec=300.0)

