"""Automated tests for ONNX Runtime object detection and Raspberry Pi 4 CPU compatibility.

Covers the 11 required migration test areas:
1. ONNX model loading
2. Missing model handling
3. Input tensor preparation (letterbox, monochrome, normalization)
4. Output tensor decoding
5. Non-maximum suppression (class-aware)
6. Bounding box coordinate restoration
7. Confidence filtering
8. Class mapping (metadata & fallback)
9. DetectionResult compatibility
10. Depth-disabled startup without PyTorch
11. Existing navigation & guidance pipeline compatibility
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from config.settings import AlertConfig, AppConfig, DepthConfig, ModelConfig, PositionConfig
from depth.depth_estimator import DepthEstimator, DepthEstimatorError
from detection.detector import COCO_CLASSES, DetectionResult, DetectorError, ObjectDetector
from detection.filtering import DetectionFilter
from detection.nearest_object import NearestObjectManager
from detection.position import HorizontalPosition, classify_horizontal_position, format_position_announcement


# ---------------------------------------------------------------------------
# Test 1: ONNX model loading
# ---------------------------------------------------------------------------
def test_onnx_model_loading():
    """Verify loading the exported ONNX model via ONNX Runtime CPU backend."""
    cfg = ModelConfig(
        model_name="models/yolov8n.onnx",
        fallback_model_name="models/yolov8n.onnx",
        backend="onnx",
        device="cpu",
        inference_size=320,
    )
    detector = ObjectDetector(cfg)
    detector.load()

    assert detector.is_loaded is True
    assert detector.active_backend == "onnx"
    assert detector.device == "cpu"
    assert len(detector.names) == 80
    assert detector.names[0] == "person"
    assert detector.names[5] == "bus"

    detector.close()
    assert detector.is_loaded is False


# ---------------------------------------------------------------------------
# Test 2: Missing model handling
# ---------------------------------------------------------------------------
def test_missing_model_handling():
    """Verify that detector fails clearly without downloading when models are missing."""
    cfg = ModelConfig(
        model_name="nonexistent_model_12345.onnx",
        fallback_model_name="nonexistent_fallback_67890.onnx",
        weights_dir="models",
        backend="onnx",
    )
    detector = ObjectDetector(cfg)
    with pytest.raises(DetectorError) as exc_info:
        detector.load()

    err_msg = str(exc_info.value)
    assert "nonexistent_model_12345.onnx" in err_msg
    assert "nonexistent_fallback_67890.onnx" in err_msg


# ---------------------------------------------------------------------------
# Test 3: Input tensor preparation (letterbox, monochrome OV9281, normalization)
# ---------------------------------------------------------------------------
def test_input_tensor_preparation():
    """Verify letterboxing, channel normalization, and monochrome sensor support."""
    cfg = ModelConfig(
        model_name="models/yolov8n.onnx",
        backend="onnx",
        inference_size=320,
    )
    detector = ObjectDetector(cfg)

    # Case A: Standard 3-channel BGR 640x480
    bgr_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blob, r, dw, dh, orig_w, orig_h = detector._preprocess_onnx(bgr_frame)

    assert blob.shape == (1, 3, 320, 320)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() <= blob.max() <= 1.0
    assert orig_w == 640
    assert orig_h == 480
    assert r == 320.0 / 640.0  # 0.5
    assert dw == 0.0
    assert dh == (320 - 240) / 2.0  # 40.0

    # Case B: 2D Monochrome / Grayscale from OV9281 sensor (480x640)
    mono_frame = np.ones((480, 640), dtype=np.uint8) * 128
    blob_m, r_m, dw_m, dh_m, orig_wm, orig_hm = detector._preprocess_onnx(mono_frame)

    assert blob_m.shape == (1, 3, 320, 320)
    assert blob_m.dtype == np.float32
    assert orig_wm == 640
    assert orig_hm == 480

    # Case C: 1-channel 3D array (480, 640, 1)
    mono_3d = np.ones((480, 640, 1), dtype=np.uint8) * 200
    blob_1c, _, _, _, _, _ = detector._preprocess_onnx(mono_3d)
    assert blob_1c.shape == (1, 3, 320, 320)

    # Case D: 4-channel BGRA array
    bgra = np.zeros((480, 640, 4), dtype=np.uint8)
    blob_4c, _, _, _, _, _ = detector._preprocess_onnx(bgra)
    assert blob_4c.shape == (1, 3, 320, 320)


# ---------------------------------------------------------------------------
# Test 4: Output tensor decoding
# ---------------------------------------------------------------------------
def test_output_tensor_decoding():
    """Verify decoding raw YOLOv8 output tensor (1, 84, 2100) into candidate detections."""
    cfg = ModelConfig(
        model_name="models/yolov8n.onnx",
        backend="onnx",
        confidence_threshold=0.40,
        iou_threshold=0.45,
        inference_size=320,
    )
    detector = ObjectDetector(cfg)
    detector._is_loaded = True
    detector._backend = "onnx"
    detector._names = {0: "person", 5: "bus"}

    # Mock ONNX InferenceSession
    mock_session = MagicMock()
    # Create synthetic output tensor (1, 84, 2100)
    synth_output = np.zeros((1, 84, 2100), dtype=np.float32)

    # Anchor #10: person (cls 0) at center (160, 160) size (80, 80) with conf 0.85
    synth_output[0, 0, 10] = 160.0  # cx
    synth_output[0, 1, 10] = 160.0  # cy
    synth_output[0, 2, 10] = 80.0   # w
    synth_output[0, 3, 10] = 80.0   # h
    synth_output[0, 4, 10] = 0.85   # score for class 0 (person)

    mock_session.run.return_value = [synth_output]
    detector._model = mock_session

    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    results = detector.detect(frame)

    assert len(results) == 1
    det = results[0]
    assert det.class_id == 0
    assert det.class_name == "person"
    assert pytest.approx(det.confidence, 0.01) == 0.85
    # For a square 320x320 frame, box should be (120, 120, 200, 200)
    assert pytest.approx(det.box[0], 1.0) == 120.0
    assert pytest.approx(det.box[1], 1.0) == 120.0
    assert pytest.approx(det.box[2], 1.0) == 200.0
    assert pytest.approx(det.box[3], 1.0) == 200.0


# ---------------------------------------------------------------------------
# Test 5: Non-maximum suppression (class-aware)
# ---------------------------------------------------------------------------
def test_non_maximum_suppression():
    """Verify overlapping boxes of same class are suppressed, different classes retained."""
    cfg = ModelConfig(
        model_name="models/yolov8n.onnx",
        backend="onnx",
        confidence_threshold=0.40,
        iou_threshold=0.50,
        inference_size=320,
    )
    detector = ObjectDetector(cfg)
    detector._is_loaded = True
    detector._backend = "onnx"
    detector._names = {0: "person", 5: "bus"}

    mock_session = MagicMock()
    synth_output = np.zeros((1, 84, 2100), dtype=np.float32)

    # Box 1: person (cls 0) at (160, 160, 80, 80), conf 0.90
    synth_output[0, :4, 0] = [160.0, 160.0, 80.0, 80.0]
    synth_output[0, 4, 0] = 0.90

    # Box 2: duplicate person (cls 0) at (162, 161, 81, 79), conf 0.70 (IoU ~ 0.95 -> must be suppressed)
    synth_output[0, :4, 1] = [162.0, 161.0, 81.0, 79.0]
    synth_output[0, 4, 1] = 0.70

    # Box 3: bus (cls 5) at same location (160, 160, 80, 80), conf 0.85 (different class -> must NOT be suppressed)
    synth_output[0, :4, 2] = [160.0, 160.0, 80.0, 80.0]
    synth_output[0, 4 + 5, 2] = 0.85

    mock_session.run.return_value = [synth_output]
    detector._model = mock_session

    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    results = detector.detect(frame)

    assert len(results) == 2
    classes_detected = {r.class_name for r in results}
    assert classes_detected == {"person", "bus"}

    person_det = next(r for r in results if r.class_name == "person")
    assert pytest.approx(person_det.confidence, 0.01) == 0.90


# ---------------------------------------------------------------------------
# Test 6: Bounding box coordinate restoration
# ---------------------------------------------------------------------------
def test_bounding_box_coordinate_restoration():
    """Verify coordinate scaling from padded letterbox to non-square original frame."""
    cfg = ModelConfig(
        model_name="models/yolov8n.onnx",
        backend="onnx",
        confidence_threshold=0.40,
        inference_size=320,
    )
    detector = ObjectDetector(cfg)
    detector._is_loaded = True
    detector._backend = "onnx"

    # Original frame 640x480 (w=640, h=480)
    # r = 320/640 = 0.5. new_unpad = (320, 240). dw = 0, dh = 40.
    # An object in original frame at (100, 100, 300, 300) [cx=200, cy=200, w=200, h=200]
    # In 320x320 letterbox: cx_l = 200*0.5 + 0 = 100, cy_l = 200*0.5 + 40 = 140, w_l = 100, h_l = 100.
    mock_session = MagicMock()
    synth_output = np.zeros((1, 84, 2100), dtype=np.float32)
    synth_output[0, :4, 0] = [100.0, 140.0, 100.0, 100.0]
    synth_output[0, 4, 0] = 0.80

    mock_session.run.return_value = [synth_output]
    detector._model = mock_session

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(frame)

    assert len(results) == 1
    det = results[0]
    # Restored coordinates should match original (100, 100, 300, 300)
    assert pytest.approx(det.box[0], 2.0) == 100.0
    assert pytest.approx(det.box[1], 2.0) == 100.0
    assert pytest.approx(det.box[2], 2.0) == 300.0
    assert pytest.approx(det.box[3], 2.0) == 300.0


# ---------------------------------------------------------------------------
# Test 7: Confidence filtering
# ---------------------------------------------------------------------------
def test_confidence_filtering():
    """Verify confidence filtering excludes sub-threshold predictions."""
    cfg = ModelConfig(
        model_name="models/yolov8n.onnx",
        backend="onnx",
        confidence_threshold=0.50,
        inference_size=320,
    )
    detector = ObjectDetector(cfg)
    detector._is_loaded = True
    detector._backend = "onnx"

    mock_session = MagicMock()
    synth_output = np.zeros((1, 84, 2100), dtype=np.float32)

    # Box 1: conf 0.45 (< 0.50 threshold)
    synth_output[0, :4, 0] = [100.0, 100.0, 40.0, 40.0]
    synth_output[0, 4, 0] = 0.45

    # Box 2: conf 0.65 (>= 0.50 threshold)
    synth_output[0, :4, 1] = [200.0, 200.0, 40.0, 40.0]
    synth_output[0, 4, 1] = 0.65

    mock_session.run.return_value = [synth_output]
    detector._model = mock_session

    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    results = detector.detect(frame)

    assert len(results) == 1
    assert pytest.approx(results[0].confidence, 0.01) == 0.65

    # Dynamic override
    results_override = detector.detect(frame, confidence_override=0.40)
    assert len(results_override) == 2


# ---------------------------------------------------------------------------
# Test 8: Class mapping (metadata & fallback)
# ---------------------------------------------------------------------------
def test_class_mapping():
    """Verify class mapping from metadata and fallback to COCO classes."""
    cfg = ModelConfig(
        model_name="models/yolov8n.onnx",
        backend="onnx",
        inference_size=320,
    )
    detector = ObjectDetector(cfg)

    # Fallback mapping
    assert len(detector.names) == 80
    assert detector.names[0] == "person"
    assert detector.names[56] == "chair"

    # Test custom metadata parsing
    mock_session = MagicMock()
    mock_meta = MagicMock()
    mock_meta.custom_metadata_map = {"names": "{0: 'pedestrian', 1: 'bike'}"}
    mock_session.get_modelmeta.return_value = mock_meta
    mock_input = MagicMock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 320, 320]
    mock_session.get_inputs.return_value = [mock_input]

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        detector._load_onnx(Path("models/yolov8n.onnx"))

    assert detector.names[0] == "pedestrian"
    assert detector.names[1] == "bike"


# ---------------------------------------------------------------------------
# Test 9: DetectionResult compatibility
# ---------------------------------------------------------------------------
def test_detection_result_compatibility():
    """Verify DetectionResult dataclass properties, immutability, and attribute types."""
    det = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.88,
        box=(50.0, 100.0, 250.0, 400.0),
        timestamp=1000.0,
    )

    assert det.center_x == 150.0
    assert det.center_y == 250.0
    assert det.width == 200.0
    assert det.height == 300.0
    assert det.is_closest is False
    assert det.distance_available is True
    assert det.distance is None

    # Test immutability (frozen dataclass)
    with pytest.raises(Exception):
        det.confidence = 0.99  # type: ignore


# ---------------------------------------------------------------------------
# Test 10: Depth-disabled startup without PyTorch installed
# ---------------------------------------------------------------------------
def test_depth_disabled_startup_without_pytorch():
    """Verify that starting NaviGlass with depth disabled does not import torch."""
    # Temporarily remove torch from sys.modules to simulate Pi environment without torch
    orig_torch = sys.modules.pop("torch", None)
    try:
        from depth.depth_estimator import DepthEstimator
        from config.settings import DepthConfig

        depth_cfg = DepthConfig(enabled=False)
        est = DepthEstimator(depth_cfg)

        assert est.is_loaded is False
        assert est.device == "cpu"

        # PyTorch must not be imported when depth is disabled
        assert "torch" not in sys.modules

        # When depth load() is called without torch installed, must raise DepthEstimatorError
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(DepthEstimatorError) as exc_info:
                est.load()
            assert "PyTorch is not installed" in str(exc_info.value)

    finally:
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch


# ---------------------------------------------------------------------------
# Test 11: Existing navigation and alert pipeline compatibility
# ---------------------------------------------------------------------------
def test_existing_navigation_compatibility():
    """Verify ONNX DetectionResults integrate cleanly into downstream navigation and alert modules."""
    alert_cfg = AlertConfig(
        eligible_classes=["person", "chair", "bus"],
        cooldown_seconds=3.0,
        announcement_interval_seconds=3.0,
        frame_width=640,
        position=PositionConfig(
            left_threshold=0.35,
            right_threshold=0.65,
            phrase_left="on your left",
            phrase_center="in front of you",
            phrase_right="on your right",
        ),
    )
    depth_cfg = DepthConfig(enabled=False)

    filter_mgr = DetectionFilter(alert_cfg, min_confidence=0.45)
    nearest_mgr = NearestObjectManager(alert_cfg, depth_cfg)

    # Create synthetic ONNX detections at left, center, right
    now = time.time()
    raw_detections = [
        # Left object (center_x = 100 -> relative_x = 100/640 = 0.156 < 0.35)
        DetectionResult(class_id=0, class_name="person", confidence=0.85, box=(50.0, 100.0, 150.0, 400.0), timestamp=now),
        # Center object (center_x = 320 -> relative_x = 320/640 = 0.50)
        DetectionResult(class_id=5, class_name="bus", confidence=0.80, box=(220.0, 50.0, 420.0, 350.0), timestamp=now),
        # Right object (center_x = 550 -> relative_x = 550/640 = 0.859 > 0.65)
        DetectionResult(class_id=0, class_name="person", confidence=0.75, box=(500.0, 100.0, 600.0, 400.0), timestamp=now),
        # Ineligible object (bottle, conf 0.90)
        DetectionResult(class_id=39, class_name="bottle", confidence=0.90, box=(10.0, 10.0, 40.0, 80.0), timestamp=now),
    ]

    # Step 1: Filter eligible classes
    filtered = filter_mgr.filter_detections(raw_detections)
    assert len(filtered) == 3
    assert all(d.class_name in ("person", "bus") for d in filtered)

    # Step 2: Position classification using horizontal classification helper
    pos_0, _ = classify_horizontal_position(filtered[0].box, frame_width=640)
    pos_1, _ = classify_horizontal_position(filtered[1].box, frame_width=640)
    pos_2, _ = classify_horizontal_position(filtered[2].box, frame_width=640)

    assert pos_0 == HorizontalPosition.LEFT
    assert pos_1 == HorizontalPosition.CENTER
    assert pos_2 == HorizontalPosition.RIGHT

    # Step 3: Nearest object manager without depth (2D position guidance mode)
    final_dets, nearest_det = nearest_mgr.process_frame(filtered, current_time=now)
    assert nearest_det is not None
    assert nearest_det.is_closest is True

    # Step 4: Spoken alert generation
    alert_phrase = nearest_mgr.format_nearest_voice_alert(nearest_det)
    assert alert_phrase is not None
    assert any(term in alert_phrase.lower() for term in ("left", "front", "right"))

    # Step 5: Direct DetectionFilter audio alert evaluation
    alert_res = filter_mgr.evaluate_alert([filtered[0]], current_time=now, frame_width=640)
    assert alert_res is not None
    assert "left" in alert_res[0].lower()
