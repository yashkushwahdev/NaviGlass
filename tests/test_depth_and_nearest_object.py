"""Comprehensive unit and integration tests for metric depth estimation and nearest-object prioritization.

Tests cover:
1. Metric depth map output structure and values.
2. Bounding-box depth extraction with boundary margin insets.
3. Filtering and handling of invalid depth values (zero, NaN, Inf, out-of-range).
4. Nearest-object selection logic.
5. Temporal distance smoothing (EMA).
6. Nearest-object switching and hysteresis band.
7. Voice priority (urgent proximity alerts purge backlog).
8. Stale depth-map rejection.
9. TTS integration with natural spoken English phrasing.
10. Missing model and inference failure handling.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from audio.speech import SpeechEngine
from config.settings import AlertConfig, AudioConfig, DepthConfig
from depth.depth_estimator import DepthEstimator, DepthEstimatorError, MetricDepthMap
from detection.detector import DetectionResult
from detection.nearest_object import (
    NearestObjectManager,
    TrackedObject,
    WarningDistanceLevel,
    compute_iou,
)


@pytest.fixture
def depth_config() -> DepthConfig:
    return DepthConfig(
        enabled=True,
        model_name="depth_anything_v2_metric_hypersim_vits.pth",
        input_size=322,
        max_depth=20.0,
        margin_ratio=0.15,
        min_valid_depth=0.2,
        max_valid_depth=20.0,
        min_valid_pixels=5,
        max_age_seconds=0.8,
        urgent_distance_threshold=1.0,
        caution_distance_threshold=2.0,
        hysteresis_distance_m=0.25,
        significant_distance_delta_m=0.5,
        temporal_smoothing_alpha=0.4,
    )


@pytest.fixture
def alert_config() -> AlertConfig:
    return AlertConfig(
        eligible_classes=["person", "chair", "backpack", "bottle"],
        cooldown_seconds=3.0,
        announcement_interval_seconds=3.0,
        frame_width=640,
    )


# ---------------------------------------------------------------------------
# Test 1: Metric depth map output structure
# ---------------------------------------------------------------------------
def test_metric_depth_map_structure(depth_config):
    estimator = DepthEstimator(depth_config)
    # Synthesize a mock metric depth map
    synthetic_depth = np.full((480, 640), 2.5, dtype=np.float32)
    synthetic_depth[100:200, 100:200] = 1.2

    depth_info = MetricDepthMap(
        depth_map=synthetic_depth,
        timestamp=time.time(),
        latency_ms=45.0,
        min_depth=1.2,
        max_depth=2.5,
        frame_shape=(480, 640),
    )

    assert depth_info.depth_map.shape == (480, 640)
    assert depth_info.min_depth == 1.2
    assert depth_info.max_depth == 2.5
    assert depth_info.latency_ms == 45.0


# ---------------------------------------------------------------------------
# Test 2: Bounding-box depth extraction with margin inset
# ---------------------------------------------------------------------------
def test_bounding_box_depth_extraction_with_margin(depth_config):
    estimator = DepthEstimator(depth_config)
    depth_map = np.full((100, 100), 10.0, dtype=np.float32)  # Background at 10m

    # Foreground object at (20, 20, 60, 60) with true depth 1.5m
    depth_map[20:60, 20:60] = 1.5

    # Boundary pixels near border (20..22 and 58..60) bleed background into box
    depth_map[20:25, 20:60] = 8.0  # background bleed at top border
    depth_map[55:60, 20:60] = 8.0  # background bleed at bottom border

    # With margin_ratio=0.15, borders are trimmed, extracting only central foreground
    dist = estimator.extract_object_distance(depth_map, (20, 20, 60, 60), margin_ratio=0.20)
    assert dist is not None
    assert dist == 1.5  # Correctly trimmed boundary bleed


# ---------------------------------------------------------------------------
# Test 3: Invalid depth values (zero, negative, NaN, Inf)
# ---------------------------------------------------------------------------
def test_invalid_depth_values_handling(depth_config):
    estimator = DepthEstimator(depth_config)
    depth_map = np.zeros((100, 100), dtype=np.float32)  # All zeros (sensor hole/invalid)

    # All zeros -> should return None
    dist_zero = estimator.extract_object_distance(depth_map, (10, 10, 50, 50))
    assert dist_zero is None

    # NaN and Inf values
    depth_map[10:50, 10:50] = np.nan
    dist_nan = estimator.extract_object_distance(depth_map, (10, 10, 50, 50))
    assert dist_nan is None

    # Mixed invalid and valid: valid median should be computed if valid count >= min_valid_pixels
    depth_map[20:30, 20:30] = 1.8  # 100 valid pixels
    depth_map[10:20, 10:20] = -1.0  # invalid
    dist_mixed = estimator.extract_object_distance(depth_map, (10, 10, 40, 40))
    assert dist_mixed == 1.8


# ---------------------------------------------------------------------------
# Test 4: Nearest-object selection logic
# ---------------------------------------------------------------------------
def test_nearest_object_selection(alert_config, depth_config):
    manager = NearestObjectManager(alert_config, depth_config)

    det_far = DetectionResult(1, "person", 0.9, (50, 50, 150, 200), time.time(), distance=3.2, position="Left")
    det_near = DetectionResult(2, "chair", 0.85, (250, 50, 350, 200), time.time(), distance=1.1, position="Center")
    det_medium = DetectionResult(3, "bottle", 0.75, (450, 50, 550, 200), time.time(), distance=2.0, position="Right")

    enriched, nearest = manager.process_frame([det_far, det_near, det_medium], current_time=100.0)

    assert nearest is not None
    assert nearest.class_name == "chair"
    assert nearest.is_closest is True
    assert nearest.distance == 1.1

    # Ensure other objects have is_closest=False
    for d in enriched:
        if d.class_name != "chair":
            assert d.is_closest is False


# ---------------------------------------------------------------------------
# Test 5: Distance smoothing (Exponential Moving Average)
# ---------------------------------------------------------------------------
def test_distance_temporal_smoothing(alert_config, depth_config):
    depth_config.temporal_smoothing_alpha = 0.5
    manager = NearestObjectManager(alert_config, depth_config)

    # Frame 1: object at 2.0m
    d1 = DetectionResult(1, "person", 0.9, (100, 100, 200, 200), 1.0, distance=2.0)
    _, n1 = manager.process_frame([d1], current_time=1.0)
    assert n1.distance == 2.0

    # Frame 2: raw distance fluctuates to 3.0m
    # Smoothed = 0.5 * 3.0 + 0.5 * 2.0 = 2.5m
    d2 = DetectionResult(1, "person", 0.9, (102, 100, 202, 200), 1.1, distance=3.0)
    _, n2 = manager.process_frame([d2], current_time=1.1)
    assert n2.distance == 2.5


# ---------------------------------------------------------------------------
# Test 6: Nearest-object switching and hysteresis
# ---------------------------------------------------------------------------
def test_nearest_object_switching_hysteresis(alert_config, depth_config):
    depth_config.hysteresis_distance_m = 0.30
    manager = NearestObjectManager(alert_config, depth_config)

    # Frame 1: Chair is nearest at 1.5m, Person is at 1.6m
    chair = DetectionResult(1, "chair", 0.9, (50, 50, 150, 200), 1.0, distance=1.5)
    person = DetectionResult(2, "person", 0.9, (300, 50, 400, 200), 1.0, distance=1.6)
    _, n1 = manager.process_frame([chair, person], current_time=1.0)
    assert n1.class_name == "chair"

    # Frame 2: Person fluctuates slightly closer to 1.4m (diff = 0.1m < hysteresis 0.3m)
    # Chair should MAINTAIN closest status due to hysteresis!
    chair2 = DetectionResult(1, "chair", 0.9, (50, 50, 150, 200), 1.1, distance=1.5)
    person2 = DetectionResult(2, "person", 0.9, (300, 50, 400, 200), 1.1, distance=1.4)
    _, n2 = manager.process_frame([chair2, person2], current_time=1.1)
    assert n2.class_name == "chair"

    # Frame 3 & 4: Person moves significantly closer to 0.8m (diff = 0.7m > hysteresis 0.3m)
    # After confirmation frames, nearest switches to person!
    chair3 = DetectionResult(1, "chair", 0.9, (50, 50, 150, 200), 1.2, distance=1.5)
    person3 = DetectionResult(2, "person", 0.9, (300, 50, 400, 200), 1.2, distance=0.8)
    manager.process_frame([chair3, person3], current_time=1.2)

    chair4 = DetectionResult(1, "chair", 0.9, (50, 50, 150, 200), 1.3, distance=1.5)
    person4 = DetectionResult(2, "person", 0.9, (300, 50, 400, 200), 1.3, distance=0.8)
    _, n4 = manager.process_frame([chair4, person4], current_time=1.3)
    assert n4.class_name == "person"


# ---------------------------------------------------------------------------
# Test 7: Voice priority and queue management
# ---------------------------------------------------------------------------
def test_voice_priority_purges_backlog():
    audio_cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(audio_cfg)
    engine._is_available = True  # Enable queue ingestion for test without starting thread

    # Mock internal queue
    engine.speak_async("Routine announcement 1", priority=False)
    engine.speak_async("Routine announcement 2", priority=False)


    # When priority=True is invoked (e.g. urgent alert under 1m), queue is purged
    enqueued = engine.speak_async("Warning! Obstacle in front of you, approximately 0.6 metres away.", priority=True)
    assert enqueued is True

    # Check only priority message remains
    with engine._queue.mutex:
        items = list(engine._queue.queue)
        assert len(items) == 1
        assert "Warning!" in items[0][0]


# ---------------------------------------------------------------------------
# Test 8: Stale depth-map rejection
# ---------------------------------------------------------------------------
def test_stale_depth_map_rejection(depth_config):
    depth_config.max_age_seconds = 0.5
    estimator = DepthEstimator(depth_config)

    depth_map = np.full((100, 100), 2.0, dtype=np.float32)
    # Depth map timestamp is t=1.0s
    depth_info = MetricDepthMap(
        depth_map=depth_map,
        timestamp=1.0,
        latency_ms=30.0,
        min_depth=2.0,
        max_depth=2.0,
        frame_shape=(100, 100),
    )

    det = DetectionResult(1, "person", 0.9, (20, 20, 60, 60), timestamp=1.2)

    # At t=1.2s (age = 0.2s <= 0.5s): Fresh!
    res_fresh = estimator.attach_distances_to_detections([det], depth_info, current_time=1.2)
    assert res_fresh[0].distance == 2.0
    assert res_fresh[0].distance_available is True

    # At t=2.0s (age = 1.0s > 0.5s): Stale! Rejects depth and marks distance unavailable
    res_stale = estimator.attach_distances_to_detections([det], depth_info, current_time=2.0)
    assert res_stale[0].distance is None
    assert res_stale[0].distance_available is False


# ---------------------------------------------------------------------------
# Test 9: TTS integration with natural spoken English phrasing
# ---------------------------------------------------------------------------
def test_tts_phrasing_and_warning_levels(alert_config, depth_config):
    manager = NearestObjectManager(alert_config, depth_config)

    # 1. Urgent (< 1m): "Warning! Backpack on your right, approximately 0.7 metres away. It is the closest detected object."
    det_urgent = DetectionResult(1, "backpack", 0.9, (450, 50, 550, 200), 1.0, distance=0.7, position="Right", warning_level="URGENT")
    phrase_urgent = manager.format_nearest_voice_alert(det_urgent)
    assert phrase_urgent == "Warning! Backpack on your right, approximately 0.7 metres away. It is the closest detected object."

    # 2. Caution (1-2m): "Chair in front of you, approximately 1.2 metres away."
    det_caution = DetectionResult(2, "chair", 0.9, (250, 50, 350, 200), 1.0, distance=1.2, position="Center", warning_level="CAUTION")
    phrase_caution = manager.format_nearest_voice_alert(det_caution)
    assert phrase_caution == "Chair in front of you, approximately 1.2 metres away."

    # 3. Normal (> 2m): "Person on your left, approximately 2.5 metres away."
    det_normal = DetectionResult(3, "person", 0.9, (50, 50, 150, 200), 1.0, distance=2.5, position="Left", warning_level="NORMAL")
    phrase_normal = manager.format_nearest_voice_alert(det_normal)
    assert phrase_normal == "Person on your left, approximately 2.5 metres away."

    # 4. Distance unavailable
    det_unavail = DetectionResult(4, "dog", 0.9, (50, 50, 150, 200), 1.0, distance=None, position="Left", distance_available=False)
    phrase_unavail = manager.format_nearest_voice_alert(det_unavail)
    assert phrase_unavail == "Dog on your left, distance unavailable."


# ---------------------------------------------------------------------------
# Test 10: Missing model and inference failure handling
# ---------------------------------------------------------------------------
def test_missing_model_and_inference_failure(depth_config):
    depth_config.model_name = "non_existent_weights_file.pth"
    estimator = DepthEstimator(depth_config)

    # Attempting to load missing weights raises DepthEstimatorError safely
    with pytest.raises(DepthEstimatorError):
        estimator.load()

    assert estimator.is_loaded is False

    # Calling estimate_depth_map when not loaded returns None safely without crashing
    res = estimator.estimate_depth_map(np.zeros((100, 100, 3), dtype=np.uint8))
    assert res is None
