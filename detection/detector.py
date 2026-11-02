"""Object detection module supporting ONNX Runtime (CPU/RPi4) and Ultralytics YOLO backends.

Preserves existing public interfaces: DetectionResult, DetectorError, and ObjectDetector.
Optimized for Raspberry Pi 4 CPU execution via ONNX Runtime without PyTorch or CUDA dependencies.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from config.settings import ModelConfig
from utils.logging_config import get_logger

logger = get_logger("detector")

# Standard 80-class COCO mapping as robust fallback if ONNX metadata is absent
COCO_CLASSES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


@dataclass(frozen=True)
class DetectionResult:
    """Represents a single detected object in an image frame."""
    class_id: int
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    timestamp: float
    distance: Optional[float] = None
    position: Optional[str] = None
    relative_x: Optional[float] = None
    is_closest: bool = False
    warning_level: Optional[str] = None
    distance_available: bool = True

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])


class DetectorError(Exception):
    """Raised when the object detection model fails to load or infer."""


class ObjectDetector:
    """Object detector supporting ONNX Runtime (CPU) and Ultralytics YOLO backends."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._active_model_name: Optional[str] = None
        self._is_loaded = False
        self._device = self._resolve_device(config.device)
        self._backend: str = "unknown"
        self._input_name: str = "images"
        self._input_shape: Tuple[int, int] = (config.inference_size, config.inference_size)
        self._names: Dict[int, str] = dict(COCO_CLASSES)

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        """Determines execution device: local CPU or optional NVIDIA CUDA GPU."""
        req = (requested_device or "cpu").lower().strip()
        if req in ("cuda", "gpu", "0"):
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(0)
                    logger.info("NVIDIA CUDA GPU acceleration enabled: %s", name)
                    return "cuda:0"
                else:
                    logger.warning("CUDA requested but no CUDA device detected. Running on CPU.")
                    return "cpu"
            except Exception as e:
                logger.warning("CUDA initialization note: %s. Using CPU.", e)
                return "cpu"
        elif req == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Auto device resolved to NVIDIA CUDA GPU: %s", torch.cuda.get_device_name(0))
                    return "cuda:0"
            except Exception:
                pass
            return "cpu"
        return "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def active_model_name(self) -> Optional[str]:
        return self._active_model_name

    @property
    def active_backend(self) -> str:
        return self._backend

    @property
    def names(self) -> Dict[int, str]:
        return dict(self._names)

    def _determine_backend(self, model_name: str) -> str:
        """Determines whether to use ONNX Runtime or Ultralytics."""
        configured = getattr(self.config, "backend", "auto").lower()
        if configured in ("onnx", "onnxruntime"):
            return "onnx"
        if configured == "ultralytics":
            return "ultralytics"

        # Auto detection based on file extension
        suffix = Path(model_name).suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        if suffix == ".pt":
            return "ultralytics"

        # If extension ambiguous, prefer onnx if onnxruntime is installed and ultralytics isn't
        try:
            import onnxruntime  # noqa: F401
            has_ort = True
        except ImportError:
            has_ort = False

        try:
            import ultralytics  # noqa: F401
            has_ultra = True
        except ImportError:
            has_ultra = False

        if has_ort and not has_ultra:
            return "onnx"
        return "ultralytics"

    def _resolve_model_path(self, model_filename: str) -> Path:
        """Resolves model path checking configured directory and project roots."""
        p_raw = Path(model_filename)
        if p_raw.is_absolute() and p_raw.exists():
            return p_raw
        if p_raw.exists():
            return p_raw

        weights_dir = Path(self.config.weights_dir)
        if len(p_raw.parts) > 1 and p_raw.parts[0] == weights_dir.name:
            p = p_raw
        else:
            p = weights_dir / model_filename

        if p.exists():
            return p

        alt_root = Path(__file__).resolve().parent.parent / p
        if alt_root.exists():
            return alt_root

        alt_direct = Path(__file__).resolve().parent.parent / weights_dir / model_filename
        if alt_direct.exists():
            return alt_direct

        return p

    def load(self) -> None:
        """Loads the YOLO detection model into memory.

        Attempts primary model first. If loading fails, attempts fallback model.
        Supports both ONNX Runtime (CPU) and Ultralytics backends.
        """
        primary_target = self.config.model_name
        fallback_target = self.config.fallback_model_name

        backend = self._determine_backend(primary_target)
        logger.info("Initializing detector with backend: %s for %s", backend, primary_target)

        primary_path = self._resolve_model_path(primary_target)
        loaded = False
        load_error = None

        if primary_path.exists():
            try:
                if backend == "onnx":
                    self._load_onnx(primary_path)
                else:
                    self._load_ultralytics(primary_path)
                self._active_model_name = primary_target
                self._backend = backend
                loaded = True
                logger.info("Primary model '%s' loaded successfully via %s.", primary_target, backend)
            except Exception as e:
                logger.warning("Primary model '%s' failed to load: %s", primary_target, e)
                load_error = e
        else:
            logger.warning("Primary model file '%s' does not exist at %s", primary_target, primary_path)
            load_error = FileNotFoundError(f"Model file not found: {primary_path}")

        if not loaded:
            # Attempt fallback model
            fb_backend = self._determine_backend(fallback_target)
            fallback_path = self._resolve_model_path(fallback_target)
            logger.info("Attempting fallback to '%s' via %s...", fallback_target, fb_backend)

            if not fallback_path.exists():
                raise DetectorError(
                    f"Both primary model '{primary_target}' and fallback model '{fallback_target}' are missing or failed to load. "
                    f"Primary error: {load_error}. Check local models directory."
                )

            try:
                if fb_backend == "onnx":
                    self._load_onnx(fallback_path)
                else:
                    self._load_ultralytics(fallback_path)
                self._active_model_name = fallback_target
                self._backend = fb_backend
                loaded = True
                logger.info("Fallback model '%s' loaded successfully via %s.", fallback_target, fb_backend)
            except Exception as fb_err:
                raise DetectorError(
                    f"Both primary model '{primary_target}' and fallback model '{fallback_target}' failed to load. "
                    f"Fallback error: {fb_err}"
                ) from fb_err

        self._is_loaded = True

        # Warmup model graph on dummy frame
        try:
            target_h, target_w = self._input_shape
            dummy = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            self.detect(dummy)
            logger.info("Model warmup complete on %s (%s).", self._device, self._backend)
        except Exception as w_err:
            logger.debug("Model warmup note: %s", w_err)

    def _load_onnx(self, model_path: Path) -> None:
        """Loads ONNX model using ONNX Runtime with CPU optimizations."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise DetectorError(
                "onnxruntime package is not installed. Install via: pip install onnxruntime"
            ) from e

        # Configure session options for Raspberry Pi 4 CPU execution
        session_options = ort.SessionOptions()
        # Raspberry Pi 4 has 4 Cortex-A72 cores; configure intra-op threads
        num_threads = int(getattr(self.config, "num_threads", 4))
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info(
            "Creating ONNX Runtime session for %s with %d threads on CPU...",
            model_path.name,
            num_threads,
        )

        try:
            self._model = ort.InferenceSession(
                str(model_path),
                session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise DetectorError(f"Failed to create ONNX Runtime session: {e}") from e

        # Inspect input metadata
        inputs = self._model.get_inputs()
        if not inputs:
            raise DetectorError("ONNX model has no input tensors.")
        self._input_name = inputs[0].name
        in_shape = inputs[0].shape  # e.g. [1, 3, 320, 320]

        if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
            self._input_shape = (in_shape[2], in_shape[3])
        else:
            sz = int(self.config.inference_size)
            self._input_shape = (sz, sz)

        logger.info(
            "ONNX model input '%s' shape: %s (using %dx%d)",
            self._input_name,
            in_shape,
            self._input_shape[1],
            self._input_shape[0],
        )

        # Inspect custom metadata for class names mapping
        try:
            meta = self._model.get_modelmeta().custom_metadata_map
            if "names" in meta:
                raw_names = meta["names"]
                parsed = None
                try:
                    parsed = json.loads(raw_names)
                except Exception:
                    try:
                        parsed = ast.literal_eval(raw_names)
                    except Exception:
                        pass
                if isinstance(parsed, dict):
                    self._names = {int(k): str(v) for k, v in parsed.items()}
                    logger.info("Loaded %d class names from ONNX metadata.", len(self._names))
        except Exception as meta_err:
            logger.debug("Could not read ONNX metadata names: %s; using standard COCO classes.", meta_err)

    def _load_ultralytics(self, model_path: Path) -> None:
        """Loads YOLO model using Ultralytics."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise DetectorError(
                "Ultralytics package is not installed. Install via: pip install ultralytics"
            ) from e

        logger.info("Loading YOLO model via Ultralytics from: %s on %s...", model_path, self._device)
        self._model = YOLO(str(model_path))
        if hasattr(self._model, "names") and isinstance(self._model.names, dict):
            self._names = {int(k): str(v) for k, v in self._model.names.items()}
        sz = int(self.config.inference_size)
        self._input_shape = (sz, sz)

    def detect(
        self,
        frame: np.ndarray,
        confidence_override: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Runs object detection inference on a frame and returns detected objects."""
        if not self._is_loaded or self._model is None:
            raise DetectorError("Detector is not loaded. Call load() before detect().")

        if frame is None or frame.size == 0:
            return []

        ts = timestamp if timestamp is not None else time.time()
        conf_thresh = confidence_override if confidence_override is not None else self.config.confidence_threshold

        if self._backend == "onnx":
            return self._detect_onnx(frame, conf_thresh, ts)
        else:
            return self._detect_ultralytics(frame, conf_thresh, ts)

    def _detect_ultralytics(
        self,
        frame: np.ndarray,
        conf_thresh: float,
        ts: float,
    ) -> List[DetectionResult]:
        """Performs inference using Ultralytics backend."""
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.config.inference_size,
                conf=conf_thresh,
                iou=self.config.iou_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            logger.error("Ultralytics inference error: %s", e)
            return []

        detections: List[DetectionResult] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = first_res.names or self._names
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(coords)):
            x1, y1, x2, y2 = coords[i]
            conf = float(confs[i])
            cid = int(cls_ids[i])
            cname = str(names.get(cid, f"class_{cid}")).lower()

            detections.append(
                DetectionResult(
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    timestamp=ts,
                )
            )

        return detections

    def _preprocess_onnx(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, float, float, int, int]:
        """Applies letterbox resizing, BGR->RGB conversion, and normalization for ONNX."""
        orig_frame = frame
        # Handle monochrome / grayscale / single-channel inputs (e.g. OV9281 monochrome sensor)
        if orig_frame.ndim == 2:
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2BGR)
        elif orig_frame.ndim == 3 and orig_frame.shape[2] == 1:
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2BGR)
        elif orig_frame.ndim == 3 and orig_frame.shape[2] == 4:
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = orig_frame.shape[:2]
        target_h, target_w = self._input_shape

        # Scale ratio (new / old)
        r = min(target_h / orig_h, target_w / orig_w)
        new_unpad_w = int(round(orig_w * r))
        new_unpad_h = int(round(orig_h * r))

        # Padding (divide evenly to center image)
        dw = (target_w - new_unpad_w) / 2.0
        dh = (target_h - new_unpad_h) / 2.0

        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            resized = cv2.resize(orig_frame, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = orig_frame

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        # Letterbox with constant padding value 114 (standard YOLO gray fill)
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # BGR to RGB, transpose to CHW, normalize to 0.0-1.0
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose((2, 0, 1))
        blob = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, r, dw, dh, orig_w, orig_h

    def _detect_onnx(
        self,
        frame: np.ndarray,
        conf_thresh: float,
        ts: float,
    ) -> List[DetectionResult]:
        """Runs ONNX inference and post-processes YOLO output tensor with class-aware NMS."""
        blob, r, dw, dh, orig_w, orig_h = self._preprocess_onnx(frame)

        try:
            outputs = self._model.run(None, {self._input_name: blob})[0]
        except Exception as e:
            logger.error("ONNX model inference error: %s", e)
            return []

        # YOLOv8 output shape is typically (1, 84, 2100) -> transpose to (2100, 84)
        if outputs.ndim == 3:
            if outputs.shape[1] < outputs.shape[2]:
                preds = np.squeeze(outputs, axis=0).T  # Shape: (N, 84)
            else:
                preds = np.squeeze(outputs, axis=0)    # Shape: (N, 84)
        elif outputs.ndim == 2:
            preds = outputs
        else:
            logger.error("Unexpected ONNX output shape: %s", outputs.shape)
            return []

        if preds.shape[1] < 5:
            logger.error("Invalid output dimension: expected >= 5, got %d", preds.shape[1])
            return []

        # Slice candidate bounding boxes (cx, cy, w, h) and class probabilities
        boxes_raw = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence threshold
        mask = confidences >= conf_thresh
        if not np.any(mask):
            return []

        boxes_raw = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert cx, cy, w, h in padded coordinates back to original image space
        cx = boxes_raw[:, 0]
        cy = boxes_raw[:, 1]
        w_box = boxes_raw[:, 2]
        h_box = boxes_raw[:, 3]

        x1_raw = (cx - w_box / 2.0 - dw) / r
        y1_raw = (cy - h_box / 2.0 - dh) / r
        w_scaled = w_box / r
        h_scaled = h_box / r

        # Prepare boxes for OpenCV class-aware NMS
        # Using class coordinate offsets prevents different classes from suppressing each other
        offset_boxes = []
        confs_list = confidences.tolist()
        for i in range(len(confidences)):
            c_offset = float(class_ids[i]) * 4096.0
            offset_boxes.append([
                float(x1_raw[i] + c_offset),
                float(y1_raw[i] + c_offset),
                float(w_scaled[i]),
                float(h_scaled[i]),
            ])

        indices = cv2.dnn.NMSBoxes(
            offset_boxes,
            confs_list,
            score_threshold=float(conf_thresh),
            nms_threshold=float(self.config.iou_threshold),
        )

        detections: List[DetectionResult] = []
        if len(indices) == 0:
            return detections

        flat_indices = np.array(indices).flatten()

        for idx in flat_indices:
            cid = int(class_ids[idx])
            cname = str(self._names.get(cid, f"class_{cid}")).lower()
            conf = float(confidences[idx])

            # Clamp coordinates to original image boundaries
            bx1 = max(0.0, min(float(orig_w), float(x1_raw[idx])))
            by1 = max(0.0, min(float(orig_h), float(y1_raw[idx])))
            bx2 = max(0.0, min(float(orig_w), float(x1_raw[idx] + w_scaled[idx])))
            by2 = max(0.0, min(float(orig_h), float(y1_raw[idx] + h_scaled[idx])))

            detections.append(
                DetectionResult(
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    box=(bx1, by1, bx2, by2),
                    timestamp=ts,
                )
            )

        return detections

    def draw_overlays(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        fps: float = 0.0,
        infer_ms: float = 0.0,
        depth_ms: float = 0.0,
        depth_status: str = "",
    ) -> np.ndarray:
        """Draws clean bounding boxes, closest object badges, and telemetry overlay on frame."""
        disp = frame.copy()

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.box]

            # Custom styling for closest detected object and warning levels
            if det.is_closest:
                if det.warning_level == "URGENT":
                    box_color = (30, 40, 230)      # High-visibility Crimson Red
                    badge_color = (40, 50, 240)
                    closest_tag = "[CLOSEST !] "
                elif det.warning_level == "CAUTION":
                    box_color = (0, 140, 255)      # Vivid Amber Orange
                    badge_color = (0, 150, 255)
                    closest_tag = "[CLOSEST] "
                else:
                    box_color = (255, 170, 0)      # Electric Blue/Cyan
                    badge_color = (255, 180, 20)
                    closest_tag = "[CLOSEST] "
                thickness = 3
            else:
                box_color = (0, 230, 115)           # Standard Emerald Green
                badge_color = (0, 200, 100)
                closest_tag = ""
                thickness = 2

            cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, thickness)

            pos_badge = f" [{det.position}]" if det.position else ""
            if det.distance is not None:
                dist_badge = f" ~{det.distance:.1f}m"
            elif not det.distance_available:
                dist_badge = " [Dist N/A]"
            else:
                dist_badge = ""

            label = f"{closest_tag}{det.class_name.capitalize()} {int(det.confidence * 100)}%{pos_badge}{dist_badge}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            text_thick = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thick)
            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1

            # Badge background
            cv2.rectangle(disp, (x1, badge_y1), (x1 + text_w + 10, badge_y2), badge_color, -1)
            # Text label
            text_color = (255, 255, 255) if det.is_closest and det.warning_level == "URGENT" else (0, 0, 0)
            cv2.putText(
                disp,
                label,
                (x1 + 5, y1 - 4),
                font,
                font_scale,
                text_color,
                text_thick,
                cv2.LINE_AA,
            )

        # Performance Overlay Banner at top-left
        if self._backend == "onnx":
            dev_tag = "ONNX-CPU"
        elif "cuda" in self._device:
            dev_tag = "YOLO-GPU"
        else:
            dev_tag = "YOLO-CPU"

        depth_tag = f" | Depth: {depth_ms:4.1f}ms ({depth_status or 'Active'})" if depth_ms > 0 or depth_status else ""
        fps_text = f"FPS: {fps:4.1f} | YOLO: {infer_ms:4.1f}ms ({dev_tag}){depth_tag} | Objects: {len(detections)}"
        banner_w = min(disp.shape[1] - 16, 580 if depth_tag else 440)
        cv2.rectangle(disp, (8, 8), (8 + banner_w, 36), (20, 20, 20), -1)
        cv2.putText(
            disp,
            fps_text,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return disp

    def close(self) -> None:
        """Releases detector resources."""
        self._model = None
        self._is_loaded = False
        logger.info("Detector unloaded.")
