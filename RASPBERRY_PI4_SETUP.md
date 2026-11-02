# NaviGlass Raspberry Pi 4 Edition — Setup & Operational Guide

## Confirmed Target Platform
- **Hardware:** Raspberry Pi 4 Model B (4 GB / 8 GB RAM recommended)
- **Operating System:** Raspberry Pi OS / Debian 12 (Bookworm), 64-bit ARM64 (`aarch64`)
- **Python Version:** Python 3.11+
- **Camera Support:**
  - Raspberry Pi Camera Module (CSI interface via `picamera2`)
  - USB UVC webcams (V4L2 interface via OpenCV)
  - Pre-detected sensor: **OV9281 monochrome** (10-bit MONO @ 1280x800)
- **Audio Support:** Linux-native offline speech synthesis via `espeak-ng` or `espeak`

---

## Sensor & Camera Implementation Notes
The detected sensor is **OV9281 monochrome** (`10-bit MONO`).
- **Picamera2 Integration:** `camera/webcam.py` includes a `PiCamera2Adapter` that captures natively via DMA buffers.
- **Grayscale Normalization:** Monochrome frames (10-bit uint16 or 8-bit uint8) are automatically normalized and expanded to 3-channel BGR (`(H, W, 3)`). This ensures downstream YOLO inference, OpenCV drawing, and GUI rendering operate seamlessly without type errors or crashing.
- **Color Note:** A colour-trained YOLO model accepts 3-channel grayscale inputs, but detection accuracy is highest with standard RGB sensors (e.g., IMX219 or IMX708 Camera Module 3).

---

## Step-by-Step Environment Setup

Run the following commands on your Raspberry Pi terminal from this repository directory:

```bash
# 1. Update package lists and install system packages from Debian repositories
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev python3-opencv python3-numpy python3-yaml python3-picamera2 espeak-ng alsa-utils

# 2. Create a virtual environment that inherits system packages
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 3. Install remaining Python dependencies
pip install -r requirements-pi.txt

# 4. Run the preflight diagnostic check
python scripts/check_rpi_environment.py
```

---

## Hardware Diagnostics

Before launching the full vision system, run the individual component diagnostics:

### 1. Test Camera Capture
```bash
# Test default / Picamera2 camera capture
python scripts/test_camera.py --frames 20 --no-preview
```

### 2. Test Spoken Audio Output
```bash
# Test offline speech synthesis via espeak-ng
python scripts/test_audio.py --phrase "NaviGlass audio system ready."
```

### 3. Test Headless YOLO Inference
```bash
# Test model inference on a sample image without requiring a live camera
python scripts/test_inference.py --image bus.jpg
```

---

## Running the Application

### Headless / CLI Mode (Recommended for SSH / Smart Glasses)
```bash
python main.py --cli
```
*(On Linux, `main.py` automatically detects and uses `config.rpi4.yaml` by default).*

### Desktop GUI Mode (When HDMI Display is Connected)
```bash
python main.py --gui
```
*(If no graphical display `$DISPLAY` is detected, NaviGlass gracefully falls back to CLI mode without crashing).*

---

## Runtime Performance Profile (`config.rpi4.yaml`)
- **Camera Resolution:** 640x480 @ 15 FPS
- **Inference Size:** 320x320
- **Frame Skip:** 2 (detects objects every 2nd frame to keep CPU cool and prevent frame stalls)
- **Metric Depth:** Disabled for initial deployment profile (reduces CPU thermal load)
- **Audio:** Enabled with `espeak-ng` non-blocking asynchronous queue
