"""Unit tests for alert cooldown suppression, prioritization, and reset with position guidance."""

import pytest
from config.settings import AlertConfig
from detection.detector import DetectionResult
from detection.filtering import DetectionFilter


def make_detection(class_name: str, confidence: float = 0.90) -> DetectionResult:
    # box center = 150 px -> on 640px frame, rel_x = 0.234 -> on your left
    return DetectionResult(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        box=(100.0, 100.0, 200.0, 200.0),
        timestamp=100.0,
    )


def test_cooldown_suppression():
    cfg = AlertConfig(eligible_classes=["person", "chair"], cooldown_seconds=4.0)
    df = DetectionFilter(cfg, min_confidence=0.30)

    det = make_detection("person")

    # Time 10.0s: First announcement -> allowed
    res1 = df.evaluate_alert([det], current_time=10.0)
    assert res1 is not None
    phrase1, _ = res1
    assert phrase1 == "Person on your left."

    # Time 12.0s (2.0s later < 4.0s cooldown): Suppressed
    res2 = df.evaluate_alert([det], current_time=12.0)
    assert res2 is None

    # Time 14.5s (4.5s later >= 4.0s cooldown): Triggered again
    res3 = df.evaluate_alert([det], current_time=14.5)
    assert res3 is not None
    phrase3, _ = res3
    assert phrase3 == "Person on your left."


def test_priority_to_highest_confidence():
    cfg = AlertConfig(eligible_classes=["person", "chair"], cooldown_seconds=4.0)
    df = DetectionFilter(cfg, min_confidence=0.30)

    det_chair = make_detection("chair", confidence=0.70)
    det_person = make_detection("person", confidence=0.92)

    # Both are eligible and not on cooldown; highest confidence (person) should be announced
    res = df.evaluate_alert([det_chair, det_person], current_time=1.0)
    assert res is not None
    phrase, det = res
    assert phrase == "Person on your left."
    assert det.class_name == "person"


def test_independent_cooldown_per_class():
    cfg = AlertConfig(eligible_classes=["person", "chair"], cooldown_seconds=4.0)
    df = DetectionFilter(cfg, min_confidence=0.30)

    det_person = make_detection("person", confidence=0.85)
    det_chair = make_detection("chair", confidence=0.80)

    # Time 1.0s: Person announced
    res1 = df.evaluate_alert([det_person], current_time=1.0)
    assert res1 is not None
    assert res1[0] == "Person on your left."

    # Time 2.0s: Chair announced (independent of person cooldown)
    res2 = df.evaluate_alert([det_person, det_chair], current_time=2.0)
    assert res2 is not None
    assert res2[0] == "Chair on your left."


def test_cooldown_reset():
    cfg = AlertConfig(eligible_classes=["person"], cooldown_seconds=5.0)
    df = DetectionFilter(cfg, min_confidence=0.30)

    det = make_detection("person")
    df.evaluate_alert([det], current_time=1.0)

    # Within cooldown
    assert df.evaluate_alert([det], current_time=2.0) is None

    # Reset
    df.reset()

    # Now permitted immediately
    res = df.evaluate_alert([det], current_time=2.0)
    assert res is not None
    assert res[0] == "Person on your left."
