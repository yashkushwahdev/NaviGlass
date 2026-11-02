r"""Tests for continuous interval announcements, position changes, multi-object handling, and mute/shutdown."""

import time
import pytest
from config.settings import AlertConfig, AudioConfig, PositionConfig
from detection.detector import DetectionResult
from detection.filtering import DetectionFilter
from audio.speech import SpeechEngine


def make_box(center_x: float) -> tuple:
    # 640 width: center_x in [0, 640]
    return (center_x - 30.0, 100.0, center_x + 30.0, 300.0)


def make_det(class_name: str, center_x: float = 320.0, confidence: float = 0.90) -> DetectionResult:
    return DetectionResult(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        box=make_box(center_x),
        timestamp=0.0,
    )


# ------------------------------------------------------------------------------
# 1. Deterministic Timing Tests for Continuous Announcements
# ------------------------------------------------------------------------------
def test_repeated_announcements_at_configured_interval():
    """Confirms that a continuously visible object repeats every announcement_interval_seconds."""
    cfg = AlertConfig(
        eligible_classes=["person"],
        announcement_interval_seconds=3.0,
        cooldown_seconds=3.0,
    )
    df = DetectionFilter(cfg)
    det = make_det("person", center_x=320.0)  # Center

    # t = 0.0s: First detection -> Announced immediately
    res0 = df.evaluate_alert([det], current_time=0.0, frame_width=640)
    assert res0 is not None
    assert res0[0] == "Person in front of you."

    # t = 1.0s (< 3.0s): Suppressed
    assert df.evaluate_alert([det], current_time=1.0, frame_width=640) is None

    # t = 2.0s (< 3.0s): Suppressed
    assert df.evaluate_alert([det], current_time=2.0, frame_width=640) is None

    # t = 3.0s (== 3.0s interval): Repeated announcement
    res3 = df.evaluate_alert([det], current_time=3.0, frame_width=640)
    assert res3 is not None
    assert res3[0] == "Person in front of you."

    # t = 4.0s (< 6.0s): Suppressed
    assert df.evaluate_alert([det], current_time=4.0, frame_width=640) is None

    # t = 6.0s (next interval): Repeated announcement
    res6 = df.evaluate_alert([det], current_time=6.0, frame_width=640)
    assert res6 is not None
    assert res6[0] == "Person in front of you."


def test_no_repeated_speech_on_every_frame():
    """Simulates 30 FPS video feed for 1 second (30 frames) to ensure speech does not spam every frame."""
    cfg = AlertConfig(
        eligible_classes=["chair"],
        announcement_interval_seconds=3.0,
    )
    df = DetectionFilter(cfg)
    det = make_det("chair", center_x=100.0)  # Left

    triggers = []
    # 30 frames at 33ms intervals
    for f in range(30):
        t = f * 0.033
        res = df.evaluate_alert([det], current_time=t, frame_width=640)
        if res is not None:
            triggers.append((t, res[0]))

    # Exactly 1 announcement (at frame 0), 0 announcements for the rest of the second
    assert len(triggers) == 1
    assert triggers[0][1] == "Chair on your left."


def test_position_change_announcement():
    """Confirms that a significant position shift triggers an immediate alert after position cooldown."""
    pos_cfg = PositionConfig(
        announce_position_change=True,
        position_change_cooldown_seconds=1.0,
    )
    cfg = AlertConfig(
        eligible_classes=["person"],
        announcement_interval_seconds=5.0,  # long interval
        position=pos_cfg,
    )
    df = DetectionFilter(cfg)

    # t = 0.0s: Person on left
    det_left = make_det("person", center_x=100.0)
    res0 = df.evaluate_alert([det_left], current_time=0.0, frame_width=640)
    assert res0 is not None
    assert res0[0] == "Person on your left."

    # t = 0.5s: Person moves to center, but position change cooldown is 1.0s -> suppressed
    det_center = make_det("person", center_x=320.0)
    assert df.evaluate_alert([det_center], current_time=0.5, frame_width=640) is None

    # t = 1.2s (> 1.0s position cooldown): Position shift announced!
    res_shift = df.evaluate_alert([det_center], current_time=1.2, frame_width=640)
    assert res_shift is not None
    assert res_shift[0] == "Person in front of you."


def test_no_announcements_after_object_disappears():
    """Ensures that when an object leaves the field of view, no announcements occur."""
    cfg = AlertConfig(
        eligible_classes=["bottle"],
        announcement_interval_seconds=3.0,
    )
    df = DetectionFilter(cfg)
    df.disappearance_timeout_seconds = 1.0

    det = make_det("bottle", center_x=550.0)  # Right

    # t = 0.0s: Bottle detected on right
    res0 = df.evaluate_alert([det], current_time=0.0, frame_width=640)
    assert res0 is not None
    assert res0[0] == "Bottle on your right."

    # t = 1.0s to 4.0s: Bottle has disappeared (empty detection list)
    for t in [1.0, 2.0, 3.0, 4.0]:
        res_empty = df.evaluate_alert([], current_time=t, frame_width=640)
        assert res_empty is None

    # t = 5.0s: Bottle reappears (more than 1.0s disappearance timeout) -> Announced immediately!
    res_reappear = df.evaluate_alert([det], current_time=5.0, frame_width=640)
    assert res_reappear is not None
    assert res_reappear[0] == "Bottle on your right."


def test_multiple_object_announcements():
    """Tests concise multi-object announcements when multiple objects are visible simultaneously."""
    pos_cfg = PositionConfig(
        multi_object_announcements=True,
        max_multi_objects=3,
    )
    cfg = AlertConfig(
        eligible_classes=["person", "chair", "bottle"],
        announcement_interval_seconds=3.0,
        position=pos_cfg,
    )
    df = DetectionFilter(cfg)

    dets = [
        make_det("person", center_x=100.0),  # Left
        make_det("chair", center_x=320.0),   # Center
        make_det("bottle", center_x=550.0),  # Right
    ]

    # t = 0.0s: Combined multi-object announcement
    res0 = df.evaluate_alert(dets, current_time=0.0, frame_width=640)
    assert res0 is not None
    assert res0[0] == "Multiple objects detected: person on your left, chair in front of you, bottle on your right."

    # t = 1.5s: Suppressed
    assert df.evaluate_alert(dets, current_time=1.5, frame_width=640) is None

    # t = 3.0s: Repeated together
    res3 = df.evaluate_alert(dets, current_time=3.0, frame_width=640)
    assert res3 is not None
    assert res3[0] == "Multiple objects detected: person on your left, chair in front of you, bottle on your right."


def test_mute_and_shutdown_behavior():
    """Confirms mute and stop behavior cleans audio queue and stops speech."""
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)
    engine._is_available = True

    # Enqueue items
    assert engine.speak_async("Alert 1") is True
    assert engine.speak_async("Alert 2") is True
    assert not engine._queue.empty()

    # Muting clears queue and prevents further speaking
    engine.set_muted(True)
    assert engine._queue.empty()
    assert engine.speak_async("Alert 3") is False

    # Unmuting allows new speaking
    engine.set_muted(False)
    assert engine.speak_async("Alert 4") is True

    # Stop shuts down and clears
    engine.stop()
    assert engine._queue.empty()
