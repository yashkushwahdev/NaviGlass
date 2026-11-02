#!/usr/bin/env python3
"""Comprehensive preflight checks for NaviGlass on Raspberry Pi OS (Debian 12 Bookworm)."""

import importlib.util
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def get_os_pretty_name() -> str:
    """Reads OS pretty name from /etc/os-release without spawning subshells."""
    os_release = Path("/etc/os-release")
    if os_release.exists():
        try:
            with open(os_release, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.strip().split("=", 1)[1].strip('"\'')
        except Exception:
            pass
    return f"{platform.system()} {platform.release()}"


def get_command_output(args: list[str]) -> str:
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=10)
        return (p.stdout or p.stderr).strip()
    except Exception as exc:
        return f"Unavailable: {exc}"


def check_ram_mb() -> str:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return f"{kb // 1024} MB"
    except Exception:
        pass
    return "Unknown"


def main() -> None:
    print("==================================================")
    print("      NaviGlass Raspberry Pi 4 Preflight Check    ")
    print("==================================================")
    print(f"OS Target:       {get_os_pretty_name()}")
    print(f"Architecture:    {platform.machine()} ({platform.architecture()[0]})")
    print(f"Python Version:  {platform.python_version()}")
    print(f"Total RAM:       {check_ram_mb()}")
    print("--------------------------------------------------")

    print("[1] Python Modules:")
    modules = {
        "cv2": "OpenCV (via apt python3-opencv)",
        "numpy": "NumPy (via apt python3-numpy)",
        "yaml": "PyYAML (via apt python3-yaml)",
        "picamera2": "Picamera2 CSI camera library (via apt python3-picamera2)",
        "onnxruntime": "ONNX Runtime CPU ARM64 engine (pip)",
        "pyttsx3": "Offline TTS (optional on Linux)",
        "torch": "PyTorch CPU ARM64 (optional, only for depth estimation)",
        "ultralytics": "Ultralytics YOLO (optional, development only)",
    }
    for mod, desc in modules.items():
        found = importlib.util.find_spec(mod) is not None
        status = "[OK] FOUND" if found else "[!] MISSING"
        print(f"  {status:<12} {mod:<12} - {desc}")

    print("\n[2] System Commands & Audio:")
    cmds = {
        "rpicam-hello": "libcamera camera diagnostic tool",
        "libcamera-hello": "libcamera legacy alias",
        "espeak-ng": "Linux speech synthesizer (preferred)",
        "espeak": "Legacy speech synthesizer",
        "aplay": "ALSA audio playback utility",
    }
    for cmd, desc in cmds.items():
        loc = shutil.which(cmd)
        status = "[OK] FOUND" if loc else "[!] MISSING"
        print(f"  {status:<12} {cmd:<16} - {loc or desc}")

    print("\n[3] Camera Hardware Detection:")
    # Check V4L2 device nodes
    v4l_nodes = sorted(list(Path("/dev").glob("video*")))
    if v4l_nodes:
        print(f"  V4L2 nodes found: {', '.join(p.name for p in v4l_nodes)}")
    else:
        print("  No /dev/video* nodes detected.")

    # Check libcamera CSI sensors
    if shutil.which("rpicam-hello"):
        out = get_command_output(["rpicam-hello", "--list-cameras"])
        print(f"  rpicam-hello output:\n    {out}")
    elif shutil.which("libcamera-hello"):
        out = get_command_output(["libcamera-hello", "--list-cameras"])
        print(f"  libcamera-hello output:\n    {out}")
    else:
        print("  rpicam-hello/libcamera-hello not found in PATH.")

    print("\n[4] Configuration Validation:")
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    rpi_cfg = project_root / "config.rpi4.yaml"
    if rpi_cfg.exists():
        try:
            from config.settings import AppConfig
            cfg = AppConfig.load(rpi_cfg)
            cfg.validate()
            print(f"  [OK] config.rpi4.yaml is present and schema-valid ({rpi_cfg}).")
            print(f"       Camera: {cfg.camera.width}x{cfg.camera.height} @ {cfg.camera.fps} FPS, backend={cfg.camera.backend}")
            print(f"       Model:  {cfg.model.model_name} (inference size {cfg.model.inference_size}, skip {cfg.model.frame_skip})")
            print(f"       Depth:  {'Enabled' if cfg.depth.enabled else 'Disabled (recommended for initial profile)'}")
        except Exception as e:
            print(f"  [!] config.rpi4.yaml validation error: {e}")
    else:
        print("  [!] config.rpi4.yaml not found.")

    print("==================================================")
    print("Preflight check completed.")
    print("==================================================")


if __name__ == "__main__":
    main()
