"""Camera-based metric depth estimation using Depth Anything V2 Metric Depth.

SAFETY & ACCURACY NOTICE:
Monocular depth estimates are algorithmic approximations derived from a single 2D camera
view and are NOT guaranteed to be accurate across all environments, lighting conditions,
or materials (e.g. glass, reflective surfaces, extreme low light). Estimates are presented
as approximate indicators (e.g. 'approximately 1.5 metres') and MUST NOT be relied upon
as certified or verified distance measurements, nor for automated collision avoidance.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import DepthConfig
from detection.detector import DetectionResult
from utils.logging_config import get_logger

logger = get_logger("depth_estimator")


class DepthEstimatorError(Exception):
    """Raised when the depth estimation model fails to load or infer."""


@dataclass
class MetricDepthMap:
    """Encapsulates a metric depth map with timing and diagnostic telemetry."""
    depth_map: np.ndarray  # 2D float32 array in metres [H, W]
    timestamp: float
    latency_ms: float
    min_depth: float
    max_depth: float
    frame_shape: Tuple[int, int]  # (H, W)


class DepthEstimator:
    """Loads Depth Anything V2 Metric Depth model once and provides real-time distance estimation."""

    def __init__(self, config: DepthConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._is_loaded = False
        self._device = self._resolve_device(config.device)
        self._last_depth_map: Optional[MetricDepthMap] = None

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model weights are loaded and ready for inference."""
        return self._is_loaded

    @property
    def device(self) -> str:
        """The active PyTorch computation device ('cpu' or 'cuda')."""
        return self._device

    def _resolve_device(self, preferred: str) -> str:
        """Selects computation device with safe CPU fallback."""
        pref_lower = (preferred or "cpu").lower()
        if "cuda" in pref_lower or "auto" in pref_lower:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            if "cuda" in pref_lower:
                logger.warning("CUDA requested for depth estimation but not available. Falling back to CPU.")
            return "cpu"
        return "cpu"

    def load(self) -> None:
        """Loads the Depth Anything V2 Metric Depth model into memory once.

        Raises:
            DepthEstimatorError: If weights file is missing, corrupt, model cannot be constructed,
                                 or PyTorch is not installed.
        """
        if self._is_loaded and self._model is not None:
            logger.debug("Depth estimator model is already loaded.")
            return

        try:
            import torch
        except ImportError as e:
            raise DepthEstimatorError(
                "PyTorch is not installed. Depth estimation requires PyTorch, "
                "which is disabled on Raspberry Pi CPU without acceleration. "
                "Keep depth.enabled=false in your configuration to run NaviGlass without PyTorch."
            ) from e

        weights_dir = Path(self.config.weights_dir)
        weights_path = weights_dir / self.config.model_name
        if not weights_path.exists():
            # Check relative to project root
            root_path = Path(__file__).resolve().parent.parent / self.config.weights_dir / self.config.model_name
            if root_path.exists():
                weights_path = root_path
            else:
                msg = f"Depth model weights not found at '{weights_path}'. Download or configure correct path."
                logger.error(msg)
                raise DepthEstimatorError(msg)

        logger.info("Loading Depth Anything V2 Metric model from: %s on %s...", weights_path, self._device)

        try:
            from depth.depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }

            enc = self.config.encoder.lower()
            if enc not in model_configs:
                logger.warning("Unknown encoder '%s', defaulting to 'vits'", enc)
                enc = "vits"

            model_kwargs = dict(model_configs[enc])
            model_kwargs["max_depth"] = float(self.config.max_depth)

            self._model = DepthAnythingV2(**model_kwargs)
            state_dict = torch.load(str(weights_path), map_location=self._device)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()

            # Warm-up inference pass to compile kernels and initialize memory
            dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            with torch.no_grad():
                self._model.infer_image(dummy_frame, input_size=self.config.input_size)

            self._is_loaded = True
            logger.info("Depth Anything V2 Metric model loaded and warmed up successfully.")
        except Exception as e:
            self._model = None
            self._is_loaded = False
            msg = f"Failed to load Depth Anything V2 model: {e}"
            logger.error(msg, exc_info=True)
            raise DepthEstimatorError(msg) from e

    def estimate_depth_map(self, frame: np.ndarray) -> Optional[MetricDepthMap]:
        """Generates a full-frame metric depth map in metres.

        Args:
            frame: Input BGR image frame from camera as numpy array.

        Returns:
            MetricDepthMap with depth values in metres, or None if inference failed.
        """
        if not self._is_loaded or self._model is None:
            logger.warning("estimate_depth_map called before model is loaded.")
            return None

        if frame is None or frame.size == 0:
            return None

        try:
            import torch
        except ImportError:
            return None

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                depth_np = self._model.infer_image(frame, input_size=self.config.input_size)

            # Ensure 2D float32 representation
            if depth_np.ndim != 2:
                depth_np = np.squeeze(depth_np)

            depth_np = depth_np.astype(np.float32)

            # Clamp valid range [0, max_depth]
            np.clip(depth_np, 0.0, self.config.max_depth, out=depth_np)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            now = time.time()

            valid_mask = (depth_np > self.config.min_valid_depth) & np.isfinite(depth_np)
            min_d = float(np.min(depth_np[valid_mask])) if np.any(valid_mask) else 0.0
            max_d = float(np.max(depth_np[valid_mask])) if np.any(valid_mask) else 0.0

            depth_info = MetricDepthMap(
                depth_map=depth_np,
                timestamp=now,
                latency_ms=latency_ms,
                min_depth=min_d,
                max_depth=max_d,
                frame_shape=(frame.shape[0], frame.shape[1]),
            )
            self._last_depth_map = depth_info
            return depth_info
        except Exception as e:
            logger.error("Depth inference error: %s", e)
            return None

    def extract_object_distance(
        self,
        depth_map: np.ndarray,
        box: Tuple[float, float, float, float],
        margin_ratio: Optional[float] = None,
    ) -> Optional[float]:
        """Extracts robust representative metric distance (in metres) for a YOLO bounding box.

        Applies an inner margin inset to eliminate background bleed around box borders,
        discards zero-depth, negative, or invalid pixels, and computes the median of valid depths.

        Args:
            depth_map: 2D float array of metric depths in metres.
            box: (x1, y1, x2, y2) bounding box in pixel coordinates.
            margin_ratio: Fraction of width/height to inset from each border (default from config).

        Returns:
            Estimated distance in metres (float rounded to 2 decimals), or None if unavailable/unreliable.
        """
        if depth_map is None or depth_map.size == 0:
            return None

        h, w = depth_map.shape[:2]
        x1, y1, x2, y2 = box

        # Clip box to image boundaries
        x1 = max(0.0, min(float(x1), float(w - 1)))
        y1 = max(0.0, min(float(y1), float(h - 1)))
        x2 = max(0.0, min(float(x2), float(w)))
        y2 = max(0.0, min(float(y2), float(h)))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 2.0 or box_h <= 2.0:
            return None

        m_ratio = margin_ratio if margin_ratio is not None else self.config.margin_ratio
        inset_x = box_w * m_ratio
        inset_y = box_h * m_ratio

        # Ensure at least 1 pixel is kept if box is very small
        crop_x1 = int(round(x1 + inset_x))
        crop_x2 = int(round(x2 - inset_x))
        crop_y1 = int(round(y1 + inset_y))
        crop_y2 = int(round(y2 - inset_y))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            # Fallback to center region if inset reduced area to zero
            crop_x1 = int(x1)
            crop_x2 = int(x2)
            crop_y1 = int(y1)
            crop_y2 = int(y2)

        sub_depth = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
        if sub_depth.size == 0:
            return None

        # Filter out zero, negative, NaN, Inf, and out-of-range pixels
        valid_mask = (
            (sub_depth >= self.config.min_valid_depth)
            & (sub_depth <= self.config.max_valid_depth)
            & np.isfinite(sub_depth)
        )

        valid_pixels = sub_depth[valid_mask]
        min_pixels = self.config.min_valid_pixels
        if valid_pixels.size < min_pixels:
            logger.debug(
                "Insufficient valid depth pixels in box (%d < %d)",
                valid_pixels.size,
                min_pixels,
            )
            return None

        # Robust statistical distance using median
        median_dist = float(np.median(valid_pixels))

        if not np.isfinite(median_dist) or median_dist <= 0.0:
            return None

        return round(median_dist, 2)

    def attach_distances_to_detections(
        self,
        detections: List[DetectionResult],
        depth_info: Optional[MetricDepthMap],
        current_time: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Associates estimated metric distances with detected objects.

        If depth_info is missing, stale (older than max_age_seconds), or invalid,
        distance is left as None and distance_available is set to False.
        """
        now = current_time if current_time is not None else time.time()
        is_fresh = False

        if depth_info is not None:
            age = now - depth_info.timestamp
            if age <= self.config.max_age_seconds:
                is_fresh = True
            else:
                logger.debug("Depth map rejected as stale (age: %.2fs > %.2fs)", age, self.config.max_age_seconds)

        results: List[DetectionResult] = []
        for det in detections:
            dist: Optional[float] = None
            avail = False
            if is_fresh and depth_info is not None:
                dist = self.extract_object_distance(depth_info.depth_map, det.box)
                avail = (dist is not None)

            results.append(
                DetectionResult(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    box=det.box,
                    timestamp=det.timestamp,
                    distance=dist,
                    position=det.position,
                    relative_x=det.relative_x,
                    is_closest=det.is_closest,
                    warning_level=det.warning_level,
                    distance_available=avail,
                )
            )

        return results

    def close(self) -> None:
        """Releases depth model resources."""
        self._model = None
        self._is_loaded = False
        self._last_depth_map = None
        logger.info("Depth estimator unloaded.")
