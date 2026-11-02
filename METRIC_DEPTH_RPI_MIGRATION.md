# NaviGlass — Raspberry Pi 4 Model B (8GB) Metric Depth Migration & Calibration Plan

This technical guide outlines the migration path, hardware optimization strategies, and camera calibration procedure for transitioning the **Camera-Based Metric Depth Estimation & Nearest-Object Alert Subsystem** from Windows Desktop to a **Raspberry Pi 4 Model B (8GB RAM)** equipped with the official Raspberry Pi Camera Module (V2 or V3).

---

## 1. Hardware Architecture & Performance Considerations

| Specification | Windows Laptop (Dev Host) | Raspberry Pi 4 Model B (Target) |
| :--- | :--- | :--- |
| **Processor** | Modern x86_64 Multi-core CPU | Broadcom BCM2711, Quad-core Cortex-A72 (ARM v8) 64-bit @ 1.5–1.8 GHz |
| **RAM** | 16 GB+ DDR4/DDR5 | 8 GB LPDDR4-3200 SDRAM |
| **SIMD Accelerators** | AVX2 / AVX-512 / FMA | ARM NEON 128-bit vector engine |
| **YOLO Inference Latency** | ~22 ms (CPU) | ~80–120 ms (NCNN / ONNX INT8) |
| **Depth Anything V2 (Small)** | ~310 ms (PyTorch CPU, 322x322) | ~1.2–2.2 s (PyTorch FP32) / ~450–650 ms (ONNX INT8 / OpenVINO) |
| **Camera Interface** | USB UVC / DirectShow (OpenCV) | MIPI-CSI2 via `libcamera` / `Picamera2` |

---

## 2. Decoupled Multi-Cadence Architecture

Running Depth Anything V2 on every frame on an embedded ARM CPU would drop camera preview frame rates to unusable levels. The solution is an **asynchronous producer-consumer worker architecture**:

```
[Picamera2 MIPI-CSI] (30 FPS Frame Capture)
        │
        ├───► [YOLO Detector Thread] (~8–12 FPS on CPU / NCNN)
        │             │
        │             ▼
        │     [Bounding Boxes] ────┐
        │                          │
        │                          ▼
        └───► [Async Depth Worker] ──► [Object Distance Extraction & Tracking]
              (1–2 FPS Background)      - Bounding box margin inset (15%)
              - ONNX INT8 / OpenVINO    - Valid median depth filtering
              - Timestamped Depth Map   - Nearest object priority & hysteresis
                                        │
                                        ▼
                               [Spoken Alert Engine] (pyttsx3 / espeak-ng)
```

1. **Camera capture thread:** Runs continuously at 25–30 FPS with zero lag.
2. **YOLO detection thread:** Runs at native ARM speed (8–12 FPS).
3. **Async Depth worker thread:** Generates a new metric depth map every ~600–900 ms in the background.
4. **Distance association:** Fast median pooling across YOLO bounding boxes takes $< 2$ ms using NumPy. Latest depth maps are retained and reused as long as they are fresh ($\le 1.0$ s). Stale maps are automatically rejected.

---

## 3. Depth Anything V2 Optimization for ARM Cortex-A72

To achieve real-time responsiveness on the Raspberry Pi 4:

### A. Resolution Scaling
- The native PyTorch model runs with input sizes that are multiples of 14.
- In `config.yaml`, switch `input_size` from `322` to `252` (or `224`):
  ```yaml
  depth:
    input_size: 252       # Reduces FLOPs by ~38% compared to 322x322
    frame_interval: 6     # Runs depth every 6 camera frames
    max_age_seconds: 1.2  # Max validity period on embedded ARM
  ```

### B. ONNX Runtime & INT8 Quantization
Convert the PyTorch `.pth` checkpoint to ONNX and quantize weights to 8-bit integers using ARM NEON optimizations:
```python
# Convert PyTorch to ONNX
import torch
from depth.depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)
model.load_state_dict(torch.load("models/depth_anything_v2_metric_hypersim_vits.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 252, 252)
torch.onnx.export(
    model, dummy_input, "models/depth_anything_v2_metric_vits_252.onnx",
    input_names=["image"], output_names=["depth"],
    opset_version=17
)
```
On Raspberry Pi, install `onnxruntime` with ARM optimizations:
```bash
pip install onnxruntime
```

---

## 4. Camera Calibration & Ground-Truth Distance Validation

### Why Calibration Is Critical
Monocular depth estimation predicts depth based on learned perspective cues from synthetic indoor training datasets (Hypersim). However:
- Monocular depth networks assume specific intrinsic camera geometry (focal length $f$, optical center $(c_x, c_y)$).
- A wide-angle Raspberry Pi Camera Module V3 has a different focal length and optical distortion than a standard laptop webcam.
- **Under no circumstances should raw estimates be presented as certified ground truth without empirical calibration.**

### Empirical Calibration Procedure (Tape Measure Protocol)
To ensure reliable distance boundaries for the visually impaired user:

1. **Setup:** Place the Raspberry Pi camera on a fixed tripod at eye/chest level (1.3 m height) facing an open corridor or hallway.
2. **Reference Targets:** Place standard test objects (e.g. chair, person, backpack) at measured ground-truth distances marked on the floor:
   - $d_{\text{ground\_truth}} \in \{0.50\,\text{m}, 0.75\,\text{m}, 1.00\,\text{m}, 1.50\,\text{m}, 2.00\,\text{m}, 3.00\,\text{m}\}$.
3. **Data Recording:** Run the diagnostic extractor across 30 consecutive frames per distance and record the model's output $d_{\text{estimated}}$.
4. **Fitting the Transfer Function:** Fit a linear correction function:
   $$d_{\text{calibrated}} = \alpha \cdot d_{\text{estimated}} + \beta$$
5. Update `config.yaml` with the calibration parameters to adjust for the camera lens field of view.

---

## 5. Raspberry Pi OS Setup Commands

### Step 1: System Packages & Dependencies
```bash
sudo apt update && sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libcamera-dev \
    libcap-dev \
    espeak-ng \
    libatlas-base-dev
```

### Step 2: Virtual Environment & Packages
```bash
python3 -m venv ~/naviglass-env --system-site-packages
source ~/naviglass-env/bin/activate

pip install --upgrade pip
pip install numpy pyyaml Pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics pyttsx3
```

### Step 3: Audio (Offline TTS via `espeak-ng`)
On Linux/Raspberry Pi, `pyttsx3` uses `espeak` / `espeak-ng` automatically:
```bash
# Test audio output via 3.5mm jack or USB headphone
espeak-ng -v en "NaviGlass Raspberry Pi platform ready."
```

### Step 4: Run Prototype in Headless / Assistive Mode
```bash
cd ~/NaviGlass-Desktop
python main.py --cli --no-preview
```
