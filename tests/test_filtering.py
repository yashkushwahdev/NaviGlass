"""Unit tests for detection confidence filtering, deduplication, and class eligibility."""

import pytest
from config.settings import AlertConfig
from detection.detector import DetectionResult
from detection.filtering import DetectionFilter


def make_det(class_name: str, confidence: float = 0.85, box=(10.0, 10.0, 100.0, 100.0)) -> DetectionResult:
    return DetectionResult(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        box=box,
        timestamp=100.0,
    )


def test_confidence_filtering():
    cfg = AlertConfig()
    df = DetectionFilter(cfg, min_confidence=0.50)

    low_conf = make_det("person", confidence=0.35)
    high_conf = make_det("person", confidence=0.75)

    filtered = df.filter_detections([low_conf, high_conf])
    assert len(filtered) == 1
    assert filtered[0].confidence == 0.75


def test_eligible_class_filtering():
    cfg = AlertConfig(eligible_classes=["person", "chair"])
    df = DetectionFilter(cfg, min_confidence=0.30)

    det_person = make_det("person", 0.90)
    det_chair = make_det("chair", 0.80)
    det_airplane = make_det("airplane", 0.95)  # Not in eligible list

    filtered = df.filter_detections([det_person, det_chair, det_airplane])
    assert len(filtered) == 2
    classes = [d.class_name for d in filtered]
    assert "person" in classes
    assert "chair" in classes
    assert "airplane" not in classes


def test_deduplication_keeps_highest_confidence():
    cfg = AlertConfig()
    df = DetectionFilter(cfg, min_confidence=0.30)

    # Multiple detections of the same class in a single frame
    det1 = make_det("person", 0.60, box=(0, 0, 50, 50))
    det2 = make_det("person", 0.92, box=(100, 100, 200, 200))
    det3 = make_det("chair", 0.75, box=(50, 50, 80, 80))

    deduped = df.deduplicate([det1, det2, det3])
    assert len(deduped) == 2

    person_det = next(d for d in deduped if d.class_name == "person")
    assert person_det.confidence == 0.92


def test_format_class_name_mapping():
    cfg = AlertConfig()
    df = DetectionFilter(cfg)

    # Vehicles should map to 'Vehicle' per default mapping
    assert df.format_class_name("car") == "Vehicle"
    assert df.format_class_name("bus") == "Vehicle"
    assert df.format_class_name("truck") == "Vehicle"

    # Standard classes capitalize cleanly
    assert df.format_class_name("person") == "Person"
    assert df.format_class_name("bottle") == "Bottle"
