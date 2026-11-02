> [!NOTE]
> **STANDALONE WINDOWS APPLICATION NOTICE:**
> This file is strictly an optional reference document detailing how to port code to a separate Raspberry Pi repository (`Naviglasses`).
> The **NaviGlass-Desktop** application runs 100% locally on your Windows 10/11 laptop using laptop hardware (laptop webcam, laptop CPU/GPU, Windows SAPI5 audio).
> It does **NOT** require, connect to, or communicate with a Raspberry Pi, SSH session, IP address, GPIO pins, or ultrasonic sensors.

---

# NaviGlass: Raspberry Pi Porting Strategy & Architecture Guide

This document outlines the architectural migration path from the Windows development prototype (`NaviGlass-Desktop`) to the Raspberry Pi deployment platform (`Naviglasses`).

---

## 1. Architectural Modularity & Portability Analysis

The codebase was designed with decoupled subsystems to minimize deployment refactoring:

| Module / Component | Platform Portability | Status & Target on Raspberry Pi OS (Bookworm) |
| :--- | :--- | :--- |
| **`naviglass.safety.AlertManager`** | **100% Portable** | Pure Python logic; retains identical cooldown tracking, class priority, and suppression. |
| **`naviglass.config.AppConfig`** | **100% Portable** | Pure Python dataclasses and YAML loader; identical schema. |
| **`naviglass.detection.ObjectDetector`** | **Partially Portable** | `ultralytics` YOLO runs on ARM64 Linux, but NCNN or ONNX runtime may be required for optimal Pi 4 CPU throughput. |
| **`naviglass.camera.OpenCVCamera`** | **Platform Specific** | Windows DirectShow (`cv2.CAP_DSHOW`) must be replaced by `Picamera2` (or V4L2) for CSI ribbon cameras. |
| **`naviglass.audio.WindowsTTS`** | **Windows Specific** | Windows SAPI5 COM bindings must be replaced by Linux-native TTS (`espeak-ng`, `piper`, or `festival`). |

---

## 2. Hardware-Specific Subsystem Replacements

### 2.1 Camera: OpenCV DirectShow $\longrightarrow$ Picamera2
- **Windows Implementation:** Uses `cv2.VideoCapture(index, cv2.CAP_DSHOW)` at $640 \times 480$.
- **Raspberry Pi Implementation:**
  - Raspberry Pi OS Bookworm utilizes the `libcamera` subsystem.
  - The standard library is `picamera2`.
  - Replace `OpenCVCamera` with `PiCSIPreviewCamera` utilizing `Picamera2.capture_array()` to acquire BGR frames with zero-copy hardware DMA buffers:
    ```python
    from picamera2 import Picamera2

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    frame = picam2.capture_array()
    ```

### 2.2 Audio: Windows SAPI5 $\longrightarrow$ Linux-Native Speech (`espeak-ng` / Piper)
- **Windows Implementation:** Asynchronous thread using `pyttsx3` with Windows SAPI5 COM apartment initialization.
- **Raspberry Pi Implementation:**
  - `pyttsx3` on Linux defaults to `espeak`, but spawning external processes (`subprocess`) or using `espeak-ng` C bindings prevents GIL contention:
    ```bash
    sudo apt-get install -y espeak-ng alsa-utils
    ```
  - Alternatively, for higher voice naturalness on Pi 4, lightweight neural TTS such as **Piper TTS** (`piper-tts`) can run efficiently on ARM64.
  - Maintain the identical asynchronous queue design (`_queue` with `maxsize=3` and non-blocking `speak_async()`) to guarantee speech synthesis never starves the camera acquisition loop.

---

## 3. Raspberry Pi OS (Bookworm 64-bit) Environment & Dependencies

### 3.1 Python & System Packages
- **OS:** Raspberry Pi OS 64-bit (Debian 12 Bookworm).
- **Python Version:** Python 3.11+ (system standard in Debian Bookworm).
- **PEP 668 Managed Environment:** Bookworm enforces externally managed environments. Install via a dedicated venv:
  ```bash
  python3 -m venv --system-site-packages ~/naviglass-env
  source ~/naviglass-env/bin/activate
  ```
  *(Note: `--system-site-packages` is recommended so the pre-compiled Debian `python3-libcamera` and `python3-picamera2` modules remain accessible).*

### 3.2 Machine Learning Dependencies
- `ultralytics` and PyTorch CPU wheels for `aarch64` (`torch` and `torchvision` compiled for ARM NEON instructions).
- For optimal CPU performance on Pi 4 without GPU acceleration:
  - Benchmark standard PyTorch vs. **NCNN export** (`model.export(format='ncnn')`). NCNN provides 2x–3x faster inference on ARM Cortex-A72 cores than stock PyTorch CPU.

---

## 4. Benchmark & Performance Tuning on Raspberry Pi 4

### 4.1 Expected Performance Discrepancy
- **Windows Laptop (Intel Core / AMD Ryzen):**
  - Warm inference latency: ~15 – 20 ms
  - End-to-end loop: ~30 – 35 FPS
- **Raspberry Pi 4B (1.5 GHz Broadcom BCM2711, 4 cores):**
  - Raw PyTorch CPU inference on YOLOv8n / YOLO26n at $320 \times 320$: ~180 – 300 ms per frame (~3.5 – 5.5 FPS).
  - With NCNN / INT8 quantization: ~90 – 140 ms per frame (~7 – 11 FPS).

### 4.2 Configuration Adjustments for Pi 4
Adjust `config.yaml` on the Raspberry Pi:
```yaml
camera:
  width: 640
  height: 480
  fps: 15
  preview: false   # Disable GUI window to eliminate X11 rendering overhead

model:
  inference_size: 256    # Lower resolution to 256x256 to reduce FLOPs
  frame_skip: 2          # Process detection every 2nd or 3rd frame to conserve CPU
  device: "cpu"

alert:
  cooldown_seconds: 5.0  # Slightly longer cooldown to prevent auditory clutter
```

---

## 5. Verification Checklist Prior to Pi Deployment

- [ ] Complete Windows algorithm validation and alert logic tuning in `NaviGlass-Desktop`.
- [ ] Ensure `Naviglasses` repository remains completely clean and untouched.
- [ ] Confirm identical alert class definitions and message formats.
- [ ] Benchmark NCNN vs. PyTorch CPU on target Pi 4 hardware.
- [ ] Test ALSA / 3.5mm audio jack or Bluetooth bone-conduction headphones on the Pi.
