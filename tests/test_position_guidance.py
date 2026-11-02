"""Unit tests for Relative Object-Position Voice Guidance."""

import pytest
from config.settings import AlertConfig, PositionConfig
from detection.detector import DetectionResult
from detection.filtering import DetectionFilter
from detection.position import (
    HorizontalPosition,
    classify_horizontal_position,
    format_multi_object_announcement,
    format_position_announcement,
)


def make_det(class_name: str, x1: float, x2: float, confidence: float = 0.85, distance: float = None) -> DetectionResult:
    return DetectionResult(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        box=(x1, 100.0, x2, 300.0),
        timestamp=100.0,
        distance=distance,
    )


# ------------------------------------------------------------------------------
# 1. Horizontal Position Classification Tests
# ------------------------------------------------------------------------------
def test_position_classification_left():
    # Box center = (50 + 150) / 2 = 100 px -> 100 / 640 = 0.156 < 0.35
    pos, rel_x = classify_horizontal_position(box=(50, 100, 150, 300), frame_width=640)
    assert pos == HorizontalPosition.LEFT
    assert round(rel_x, 3) == 0.156
    assert pos.phrase == "on your left"


def test_position_classification_center():
    # Box center = (260 + 380) / 2 = 320 px -> 320 / 640 = 0.50 (in [0.35, 0.65))
    pos, rel_x = classify_horizontal_position(box=(260, 100, 380, 300), frame_width=640)
    assert pos == HorizontalPosition.CENTER
    assert round(rel_x, 3) == 0.50
    assert pos.phrase == "in front of you"


def test_position_classification_right():
    # Box center = (480 + 600) / 2 = 540 px -> 540 / 640 = 0.844 >= 0.65
    pos, rel_x = classify_horizontal_position(box=(480, 100, 600, 300), frame_width=640)
    assert pos == HorizontalPosition.RIGHT
    assert round(rel_x, 3) == 0.844
    assert pos.phrase == "on your right"


# ------------------------------------------------------------------------------
# 2. Boundary Hysteresis Tests (Prevent Rapid Speech Oscillation)
# ------------------------------------------------------------------------------
def test_boundary_hysteresis_prevents_oscillation():
    # Left threshold is 0.35, hysteresis is 0.04 -> effective exit threshold is 0.39
    # If currently LEFT, an object at rel_x = 0.36 should stay LEFT
    box_at_36 = (0.35 * 640, 100, 0.37 * 640, 300)  # center = 0.36
    pos_without_hysteresis, _ = classify_horizontal_position(box_at_36, frame_width=640)
    assert pos_without_hysteresis == HorizontalPosition.CENTER

    pos_with_hysteresis, _ = classify_horizontal_position(
        box_at_36,
        frame_width=640,
        current_position=HorizontalPosition.LEFT,
        hysteresis=0.04,
    )
    # Remains LEFT due to hysteresis buffer
    assert pos_with_hysteresis == HorizontalPosition.LEFT

    # Move far enough to center (0.42 > 0.39)
    box_at_42 = (0.40 * 640, 100, 0.44 * 640, 300)
    pos_moved, _ = classify_horizontal_position(
        box_at_42,
        frame_width=640,
        current_position=HorizontalPosition.LEFT,
        hysteresis=0.04,
    )
    assert pos_moved == HorizontalPosition.CENTER


# ------------------------------------------------------------------------------
# 3. Spoken Phrase Formatting Tests
# ------------------------------------------------------------------------------
def test_format_announcements():
    assert format_position_announcement("person", HorizontalPosition.LEFT) == "Person on your left."
    assert format_position_announcement("chair", HorizontalPosition.CENTER) == "Chair in front of you."
    assert format_position_announcement("bottle", HorizontalPosition.RIGHT) == "Bottle on your right."


def test_format_announcements_with_optional_distance():
    # In voice guidance mode, estimated distance is strictly omitted from spoken guidance
    ann_dist = format_position_announcement("person", HorizontalPosition.LEFT, distance=2.0)
    assert ann_dist == "Person on your left."

    # Distance = 1.0 meter: distance omitted
    ann_1m = format_position_announcement("chair", HorizontalPosition.CENTER, distance=1.0)
    assert ann_1m == "Chair in front of you."

    # Distance is None: clean phrase
    ann_none = format_position_announcement("bottle", HorizontalPosition.RIGHT, distance=None)
    assert ann_none == "Bottle on your right." 
def test_multi_object_announcement_format():
    items = [
        ("person", HorizontalPosition.LEFT, None),
        ("chair", HorizontalPosition.CENTER, None),
        ("bottle", HorizontalPosition.RIGHT, None),
    ]
    res = format_multi_object_announcement(items)
    assert res == "Multiple objects detected: person on your left, chair in front of you, bottle on your right."


# ------------------------------------------------------------------------------
# 4. End-to-End Alert Evaluation Integration Tests
# ------------------------------------------------------------------------------
def test_evaluate_alert_announces_correct_positions():
    cfg = AlertConfig(eligible_classes=["person", "chair", "bottle"], cooldown_seconds=4.0)
    df = DetectionFilter(cfg)

    # Person on left (center_x = 100 px)
    det_person = make_det("person", x1=50, x2=150, confidence=0.88)
    res_person = df.evaluate_alert([det_person], current_time=1.0, frame_width=640)
    assert res_person is not None
    assert res_person[0] == "Person on your left."
    assert res_person[1].position == "Left"

    # Chair in center (center_x = 320 px)
    det_chair = make_det("chair", x1=270, x2=370, confidence=0.85)
    res_chair = df.evaluate_alert([det_chair], current_time=2.0, frame_width=640)
    assert res_chair is not None
    assert res_chair[0] == "Chair in front of you."
    assert res_chair[1].position == "Center"

    # Bottle on right (center_x = 540 px)
    det_bottle = make_det("bottle", x1=490, x2=590, confidence=0.80)
    res_bottle = df.evaluate_alert([det_bottle], current_time=3.0, frame_width=640)
    assert res_bottle is not None
    assert res_bottle[0] == "Bottle on your right."
    assert res_bottle[1].position == "Right"


def test_cooldown_suppression_on_unchanged_object():
    cfg = AlertConfig(eligible_classes=["chair"], cooldown_seconds=4.0)
    df = DetectionFilter(cfg)

    det_chair = make_det("chair", x1=270, x2=370)

    # First detection
    res1 = df.evaluate_alert([det_chair], current_time=10.0, frame_width=640)
    assert res1 is not None
    assert res1[0] == "Chair in front of you."

    # Immediate next frame (10.1s): Suppressed
    res2 = df.evaluate_alert([det_chair], current_time=10.1, frame_width=640)
    assert res2 is None

    # After cooldown elapsed (14.5s): Allowed again
    res3 = df.evaluate_alert([det_chair], current_time=14.5, frame_width=640)
    assert res3 is not None
    assert res3[0] == "Chair in front of you."


def test_position_change_announcement():
    cfg = AlertConfig(eligible_classes=["person"], cooldown_seconds=5.0)
    cfg.position.announce_position_change = True
    cfg.position.position_change_cooldown_seconds = 1.0
    df = DetectionFilter(cfg)

    # Person starts on LEFT at 10.0s
    det_left = make_det("person", x1=50, x2=150)
    res1 = df.evaluate_alert([det_left], current_time=10.0, frame_width=640)
    assert res1 is not None
    assert res1[0] == "Person on your left."

    # Person moves to CENTER at 11.5s (> position_change_cooldown 1.0s, but < full cooldown 5.0s)
    det_center = make_det("person", x1=270, x2=370)
    res2 = df.evaluate_alert([det_center], current_time=11.5, frame_width=640)
    assert res2 is not None
    assert res2[0] == "Person in front of you."


def test_multi_object_alert_evaluation():
    cfg = AlertConfig(eligible_classes=["person", "chair", "bottle"])
    df = DetectionFilter(cfg)

    det_p = make_det("person", x1=50, x2=150, confidence=0.90)
    det_c = make_det("chair", x1=270, x2=370, confidence=0.85)
    det_b = make_det("bottle", x1=490, x2=590, confidence=0.80)

    res = df.evaluate_multi_alert([det_p, det_c, det_b], current_time=1.0, frame_width=640)
    assert res is not None
    msg, dets = res
    assert msg == "Multiple objects detected: person on your left, chair in front of you, bottle on your right."
    assert len(dets) == 3
