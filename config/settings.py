"""Strongly typed application configuration and schema validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ConfigurationError(Exception):
    """Raised when configuration values are missing, invalid, or corrupted."""


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    preview: bool = True
    source: str = ""
    backend: str = "auto"  # "auto", "picamera2", "opencv", "v4l2", "dshow"

    def validate(self) -> None:
        if self.index < 0:
            raise ConfigurationError(f"camera.index must be >= 0, got {self.index}")
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(f"Invalid dimensions: {self.width}x{self.height}")
        if self.fps <= 0:
            raise ConfigurationError(f"camera.fps must be > 0, got {self.fps}")
        valid_backends = {"auto", "picamera2", "opencv", "v4l2", "dshow"}
        if self.backend.lower() not in valid_backends:
            raise ConfigurationError(
                f"camera.backend must be one of {valid_backends}, got '{self.backend}'"
            )


@dataclass
class ModelConfig:
    model_name: str = "yolo26n.pt"
    fallback_model_name: str = "yolov8n.pt"
    weights_dir: str = "models"
    inference_size: int = 320
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    device: str = "cpu"
    frame_skip: int = 1
    backend: str = "auto"  # "auto", "onnx", "ultralytics"
    num_threads: int = 4   # Intra-op threads for CPU execution (e.g. ONNX Runtime)

    def validate(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ConfigurationError(f"iou_threshold must be between 0.0 and 1.0, got {self.iou_threshold}")
        if self.inference_size <= 0 or self.inference_size % 32 != 0:
            raise ConfigurationError(
                f"inference_size must be a positive multiple of 32 (e.g. 320, 640), got {self.inference_size}"
            )
        if self.frame_skip < 1:
            raise ConfigurationError(f"frame_skip must be >= 1, got {self.frame_skip}")
        valid_backends = {"auto", "onnx", "ultralytics"}
        if self.backend.lower() not in valid_backends:
            raise ConfigurationError(
                f"model.backend must be one of {valid_backends}, got '{self.backend}'"
            )
        if self.num_threads < 1:
            raise ConfigurationError(f"model.num_threads must be >= 1, got {self.num_threads}")


@dataclass
class PositionConfig:
    """Configuration for relative object position classification and voice guidance."""
    enabled: bool = True
    left_threshold: float = 0.35
    right_threshold: float = 0.65
    hysteresis: float = 0.04
    phrase_left: str = "on your left"
    phrase_center: str = "in front of you"
    phrase_right: str = "on your right"
    announce_position_change: bool = True
    position_change_cooldown_seconds: float = 1.5
    multi_object_announcements: bool = False
    max_multi_objects: int = 3

    def validate(self) -> None:
        if not (0.0 < self.left_threshold < self.right_threshold < 1.0):
            raise ConfigurationError(
                f"Position thresholds invalid: left ({self.left_threshold}) must be < right ({self.right_threshold})"
            )
        if self.hysteresis < 0.0 or self.hysteresis > 0.15:
            raise ConfigurationError(f"Hysteresis must be between 0.0 and 0.15, got {self.hysteresis}")


@dataclass
class AlertConfig:
    eligible_classes: List[str] = field(
        default_factory=lambda: [
            "person", "chair", "table", "bottle", "cup", "couch", "laptop",
            "cell phone", "book", "dog", "cat", "car", "bus", "truck",
            "bicycle", "motorcycle", "backpack", "handbag", "door"
        ]
    )
    cooldown_seconds: float = 4.0
    announcement_interval_seconds: float = 3.0
    announcement_format: str = "{class_name} {position}."
    include_position: bool = True
    enable_ocr: bool = False
    frame_width: int = 640
    position: PositionConfig = field(default_factory=PositionConfig)
    # Mapping to simplify or group spoken class names
    spoken_class_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "car": "Vehicle",
            "bus": "Vehicle",
            "truck": "Vehicle",
            "motorcycle": "Vehicle",
            "bicycle": "Bicycle",
            "cell phone": "Phone",
            "dining table": "Table",
            "potted plant": "Plant",
        }
    )

    def validate(self) -> None:
        if self.cooldown_seconds < 0.0:
            raise ConfigurationError(f"cooldown_seconds must be >= 0.0, got {self.cooldown_seconds}")
        if self.announcement_interval_seconds < 0.0:
            raise ConfigurationError(f"announcement_interval_seconds must be >= 0.0, got {self.announcement_interval_seconds}")
        self.position.validate()


@dataclass
class AudioConfig:
    enabled: bool = True
    speech_rate_wpm: int = 180
    volume: float = 1.0
    fallback_to_terminal: bool = True

    def validate(self) -> None:
        if not (50 <= self.speech_rate_wpm <= 400):
            raise ConfigurationError(f"speech_rate_wpm must be between 50 and 400, got {self.speech_rate_wpm}")
        if not (0.0 <= self.volume <= 1.0):
            raise ConfigurationError(f"volume must be between 0.0 and 1.0, got {self.volume}")


@dataclass
class DepthConfig:
    """Configuration for Depth Anything V2 metric depth estimation."""
    enabled: bool = True
    model_name: str = "depth_anything_v2_metric_hypersim_vits.pth"
    weights_dir: str = "models"
    encoder: str = "vits"
    dataset: str = "hypersim"
    max_depth: float = 20.0
    input_size: int = 322  # Multiple of 14 (e.g. 23*14=322), optimal for real-time CPU performance
    device: str = "cpu"
    frame_interval: int = 3  # Run depth inference every N frames to conserve CPU
    max_age_seconds: float = 0.8  # Reject stale depth maps
    margin_ratio: float = 0.15  # Inset ratio on bounding boxes to avoid boundary bleed
    min_valid_depth: float = 0.2
    max_valid_depth: float = 20.0
    min_valid_pixels: int = 10

    # Warning thresholds in metres
    urgent_distance_threshold: float = 1.0
    caution_distance_threshold: float = 2.0

    # Nearest object selection & tracking hysteresis
    hysteresis_distance_m: float = 0.25
    significant_distance_delta_m: float = 0.5
    temporal_smoothing_alpha: float = 0.35

    def __post_init__(self) -> None:
        try:
            self.max_depth = float(self.max_depth)
            self.input_size = int(self.input_size)
            self.frame_interval = int(self.frame_interval)
            self.max_age_seconds = float(self.max_age_seconds)
            self.margin_ratio = float(self.margin_ratio)
            self.min_valid_depth = float(self.min_valid_depth)
            self.max_valid_depth = float(self.max_valid_depth)
            self.min_valid_pixels = int(self.min_valid_pixels)
            self.urgent_distance_threshold = float(self.urgent_distance_threshold)
            self.caution_distance_threshold = float(self.caution_distance_threshold)
            self.hysteresis_distance_m = float(str(self.hysteresis_distance_m).split("#")[0].strip())
            self.significant_distance_delta_m = float(str(self.significant_distance_delta_m).split("#")[0].strip())
            self.temporal_smoothing_alpha = float(str(self.temporal_smoothing_alpha).split("#")[0].strip())
        except Exception as e:
            raise ConfigurationError(f"Invalid depth configuration parameter: {e}") from e

    def validate(self) -> None:
        if self.max_depth <= 0.0:
            raise ConfigurationError(f"depth.max_depth must be > 0.0, got {self.max_depth}")
        if self.input_size <= 0 or self.input_size % 14 != 0:
            raise ConfigurationError(f"depth.input_size must be a positive multiple of 14, got {self.input_size}")
        if self.frame_interval < 1:
            raise ConfigurationError(f"depth.frame_interval must be >= 1, got {self.frame_interval}")
        if self.max_age_seconds <= 0.0:
            raise ConfigurationError(f"depth.max_age_seconds must be > 0.0, got {self.max_age_seconds}")
        if not (0.0 <= self.margin_ratio < 0.45):
            raise ConfigurationError(f"depth.margin_ratio must be between 0.0 and 0.45, got {self.margin_ratio}")
        if self.urgent_distance_threshold >= self.caution_distance_threshold:
            raise ConfigurationError(
                f"urgent_distance_threshold ({self.urgent_distance_threshold}) must be < caution_distance_threshold ({self.caution_distance_threshold})"
            )


@dataclass
class MonitoringConfig:
    fps_window_size: int = 20
    metrics_report_interval_seconds: float = 5.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def validate(self) -> None:
        self.camera.validate()
        self.model.validate()
        self.depth.validate()
        self.alert.validate()
        self.audio.validate()

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> AppConfig:
        p = Path(config_path) if config_path else Path("config.yaml")
        if not p.exists():
            cfg = cls()
            cfg.validate()
            return cfg

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to read config file {p}: {e}") from e

        # Handle nested position config inside alert safely
        alert_dict = dict(data.get("alert", {}))
        if "position" in alert_dict and isinstance(alert_dict["position"], dict):
            alert_dict["position"] = PositionConfig(**alert_dict["position"])

        depth_dict = dict(data.get("depth", {}))

        cfg = cls(
            camera=CameraConfig(**data.get("camera", {})),
            model=ModelConfig(**data.get("model", {})),
            depth=DepthConfig(**depth_dict),
            alert=AlertConfig(**alert_dict),
            audio=AudioConfig(**data.get("audio", {})),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
        )
        cfg.validate()
        return cfg

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False, sort_keys=False)

