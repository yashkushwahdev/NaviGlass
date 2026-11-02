# NaviGlass — Raspberry Pi 4 Installation & Operational Setup Guide

This guide provides complete instructions and exact commands to deploy and run **NaviGlass** on a **Raspberry Pi 4 Model B (8 GB RAM)** running **Debian 12 Bookworm (64-bit ARM64)** with the **OV9281 monochrome camera sensor** using the **ONNX Runtime CPU backend** (without PyTorch or CUDA).

---

## 1. System Requirements & Hardware Target

- **SBC:** Raspberry Pi 4 Model B (8 GB RAM recommended, 4 GB minimum)
- **OS:** Raspberry Pi OS 64-bit (Debian 12 Bookworm, kernel 6.1+)
- **Architecture:** ARM64 (`aarch64`), Python 3.11
- **Inference Engine:** ONNX Runtime (`onnxruntime>=1.16.0`) with ARM NEON acceleration
- **Camera Module:** Raspberry Pi CSI Camera Module (detected: `OV9281 [1280x800 10-bit MONO]`) via Picamera2
- **Audio Output:** 3.5mm headphone jack, USB audio, or Bluetooth earpiece (configured via ALSA/espeak-ng)

---

## 2. Windows Model Export & Transfer Workflow

The YOLO detection model must be exported on the Windows development machine (which contains Ultralytics and PyTorch) and transferred to the Raspberry Pi. **PyTorch is not installed or required on the Raspberry Pi.**

### Step 2.1: Export YOLOv8 to ONNX on Windows
Run the reproducible export script from the project root on Windows:

```powershell
python scripts/export_yolo_onnx.py --weights models/yolov8n.pt --imgsz 320 --output models/yolov8n.onnx
```

*Export Details:*
- Source weights: `models/yolov8n.pt` (preserved, never modified)
- Destination: `models/yolov8n.onnx` (~12.1 MB)
- Input tensor: `images` shape `[1, 3, 320, 320]`, `float32` (static dimensions for predictable Pi inference)
- Output tensor: `output0` shape `[1, 84, 2100]`, `float32`
- Class metadata: 80 COCO classes embedded directly into the ONNX model graph

### Step 2.2: Transfer ONNX Model to Raspberry Pi
Transfer the generated `yolov8n.onnx` file to the Raspberry Pi using `scp`:

```powershell
scp models/yolov8n.onnx <pi_username>@<pi_ip_address>:~/NaviGlass-RaspberryPi4/models/
```

Verify that the target directory on the Pi contains `models/yolov8n.onnx`.

---

## 3. Step-by-Step Raspberry Pi Setup

### Step 3.1: Install System Prerequisites via Debian `apt`
Raspberry Pi OS Bookworm uses PEP 668 to manage Python packages. To preserve system stability and leverage hardware-accelerated drivers (Picamera2, libcamera, OpenCV), install native Debian packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-picamera2 \
    python3-pil \
    espeak-ng \
    alsa-utils
```

### Step 3.2: Create and Activate Virtual Environment
Create a virtual environment that inherits access to system-installed packages (`--system-site-packages`):

```bash
# Navigate to the project directory
cd ~/NaviGlass-RaspberryPi4

# Create virtual environment
python3 -m venv --system-site-packages .venv

# Activate the virtual environment
source .venv/bin/activate
```

*(Note: Always run `source .venv/bin/activate` whenever opening a new terminal session on the Pi).*

### Step 3.3: Install Raspberry Pi Python Dependencies
Install the lightweight runtime dependencies (ONNX Runtime, psutil, pyttsx3, pytest). **No PyTorch or Ultralytics packages are installed:**

```bash
pip install -r requirements-pi.txt
```

---

## 4. Hardware Preflight Verification

Before launching the main navigation pipeline, run the automated verification scripts:

### Step 4.1: Environment Check
```bash
python scripts/check_rpi_environment.py
```
*Expected Result:*
- `cv2`, `numpy`, `yaml`, `picamera2`, `onnxruntime` show `[OK] FOUND`.
- `espeak-ng` shows `[OK] FOUND`.
- `config.rpi4.yaml` validates successfully with `backend=onnx` and `model=yolov8n.onnx`.

### Step 4.2: Verify Spoken Audio
```bash
python scripts/test_audio.py --phrase "NaviGlass audio system online."
```
*Expected Result:* Natural voice announcement plays cleanly through the audio output.

### Step 4.3: Verify OV9281 Camera Capture
```bash
python scripts/test_camera.py --frames 20 --no-preview
```
*Expected Result:* Captures 20 frames from the OV9281 sensor, converts monochrome to 3-channel BGR, and logs FPS.

### Step 4.4: Verify ONNX Inference (Headless)
```bash
python scripts/test_inference.py --image bus.jpg --model models/yolov8n.onnx --backend onnx
```
*Expected Result:* Loads `models/yolov8n.onnx`, executes inference on CPU, and reports detected objects (`person`, `bus`).

---

## 5. Running NaviGlass on Raspberry Pi 4

### Mode A: Smart Glasses / Headless CLI Mode (Recommended)
This mode runs the entire vision and speech navigation loop in the terminal without GUI overhead:

```bash
python main.py --cli --config config.rpi4.yaml
```
*(On Linux, `main.py` defaults to `config.rpi4.yaml` automatically).*

### Mode B: Accessible Desktop GUI Mode (When Monitor is Attached)
```bash
python main.py --gui --config config.rpi4.yaml
```
*(If no desktop display server `$DISPLAY` is found, NaviGlass gracefully falls back to CLI mode without crashing).*

---

## 6. Running Automated Tests

Run the full pytest suite to verify model loading, input letterboxing, tensor decoding, NMS, and navigation logic:

```bash
pytest
```
*Expected Result:* 70 tests passed (100% pass rate).

---

## 7. Enabling Depth Estimation in the Future

In the initial Raspberry Pi 4 profile, metric depth estimation is **disabled by default** (`depth.enabled: false` in `config.rpi4.yaml`) because the Depth Anything V2 model currently depends on PyTorch.

To enable metric depth estimation in the future:
1. Export the Depth Anything V2 PyTorch model to ONNX using an ONNX export script.
2. Update `depth/depth_estimator.py` with an ONNX Runtime inference session.
3. Set `depth.enabled: true` in `config.rpi4.yaml`.
4. Perform the tape-measure camera calibration procedure documented in `METRIC_DEPTH_RPI_MIGRATION.md` to compute accurate metric scale factors for your specific camera lens.

---

## 8. Known Limitations of CPU Inference on Raspberry Pi 4

1. **CPU Inference Latency:**
   - The Raspberry Pi 4 CPU features four ARM Cortex-A72 cores running at 1.5 GHz.
   - Using static 320x320 input dimensions and 4 intra-op threads with NEON SIMD, YOLOv8n ONNX achieves ~70–120 ms latency per frame (~8–14 FPS).
   - Frame skipping (`frame_skip: 2`) is enabled in `config.rpi4.yaml` to ensure responsive audio announcements and maintain low CPU temperatures.
2. **Monochrome Sensor Color Invariance:**
   - The OV9281 is an 8-bit/10-bit monochrome global shutter sensor.
   - NaviGlass automatically replicates the single grayscale channel across 3 color channels (`cv2.COLOR_GRAY2BGR`).
   - Objects relying heavily on chromatic contrast (e.g. distinguishing green from red traffic lights) may have slightly lower detection confidence.
3. **Thermal Management:**
   - Sustained multi-threaded CPU inference will generate heat. An active cooling fan or heatsink case is strongly recommended to prevent the Pi 4 from thermal throttling (clock drops from 1.5 GHz to 1.0 GHz at 80°C).
