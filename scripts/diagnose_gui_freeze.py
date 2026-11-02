"""Script to reproduce and diagnose the GUI camera freeze in NaviGlassGUI.

Launches NaviGlassGUI in the exact same manner as `python main.py --gui`,
starts the camera, runs the Tkinter event loop, and logs all diagnostics:
- Camera frame ID vs GUI displayed frame ID
- Queue sizes
- Worker thread alive status
- Exact exception tracebacks if any thread dies
- Thread stack traces if freeze occurs
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

import tkinter as tk
from app.gui import NaviGlassGUI
from config.settings import AppConfig
from utils.logging_config import setup_logging, get_logger

logger = get_logger("gui_diagnostic")


def diagnose_gui(duration_sec: float = 30.0):
    setup_logging()
    print("=========================================================")
    print("      NaviGlass Desktop GUI Freeze Diagnostic Test       ")
    print("=========================================================")

    cfg = AppConfig.load()
    root = tk.Tk()
    app = NaviGlassGUI(root, cfg)

    # Let the window initialize and render its geometry
    for _ in range(10):
        root.update()
        time.sleep(0.05)

    print("\n[GUI INIT] Window initialized. Starting camera...")
    app.start_camera()

    t_start = time.time()
    last_log_t = t_start
    last_fid_seen = -1
    freeze_detected = False

    try:
        while time.time() - t_start < duration_sec:
            # Process Tkinter event loop
            root.update()
            time.sleep(0.01)

            now = time.time()
            if now - last_log_t >= 1.0:
                grabber_alive = app.webcam.grabber_alive
                det_alive = app._worker_thread is not None and app._worker_thread.is_alive()
                depth_alive = app._depth_thread is not None and app._depth_thread.is_alive()
                q_size = app._frame_queue.qsize()
                cam_fid = app.webcam.frame_id
                cam_fps = app.webcam.grab_fps
                succ_reads = app.webcam.successful_reads
                fail_reads = app.webcam.failed_reads
                latest_cap_ts = app.webcam.latest_capture_timestamp
                latest_disp_ts = app._latest_displayed_timestamp
                depth_busy = app._depth_busy.locked()

                cap_lag = now - latest_cap_ts if latest_cap_ts > 0 else 0.0
                disp_lag = now - latest_disp_ts if latest_disp_ts > 0 else 0.0

                print(
                    f"[{now - t_start:4.1f}s] CamFID={cam_fid:5d} | "
                    f"CamGrabFPS={cam_fps:4.1f} | QSize={q_size} | "
                    f"CapLag={cap_lag:4.2f}s | DispLag={disp_lag:4.2f}s | "
                    f"GrabberAlive={grabber_alive} | DetAlive={det_alive} | DepthAlive={depth_alive} | DepthBusy={depth_busy}"
                )

                if disp_lag > 2.0 and not freeze_detected:
                    freeze_detected = True
                    print("\n" + "!" * 60)
                    print(f"!!! FREEZE DETECTED AT {now - t_start:.2f}s !!!")
                    print(f"!!! DispLag={disp_lag:.2f}s | CapLag={cap_lag:.2f}s | CamFID={cam_fid}")
                    print("Thread stack traces:")
                    for th in threading.enumerate():
                        print(f"\n--- Thread: {th.name} (daemon={th.daemon}, alive={th.is_alive()}) ---")
                        # Print stack
                        stack = sys._current_frames().get(th.ident)
                        if stack:
                            traceback.print_stack(stack)
                    print("!" * 60 + "\n")

                last_log_t = now

    finally:
        print("\nStopping GUI and releasing resources...")
        try:
            app.stop_camera()
        except Exception as e:
            print("Error stopping camera:", e)
        try:
            root.destroy()
        except Exception:
            pass

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    diagnose_gui(duration_sec=30.0)

