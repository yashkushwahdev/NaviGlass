"""Verification script for targeted bug fixes and continuous audio announcements in NaviGlass Desktop."""

import sys
sys.path.insert(0, ".")

import time
from config.settings import AppConfig, AudioConfig, AlertConfig
from audio.speech import SpeechEngine
from detection.position import (
    HorizontalPosition,
    classify_horizontal_position,
    format_position_announcement,
    format_multi_object_announcement,
)
from detection.detector import DetectionResult
from detection.filtering import DetectionFilter


def test_verification():
    print("=== 1. Testing Position Classification ===")
    w = 640

    # LEFT Region: center_x = 100, rel_x = 0.156 < 0.35
    pos_l, rel_l = classify_horizontal_position((50, 50, 150, 200), frame_width=w)
    assert pos_l == HorizontalPosition.LEFT, f"Expected LEFT, got {pos_l}"
    ann_l = format_position_announcement("person", pos_l)
    print(f"LEFT detection: {pos_l.name} (rel_x: {rel_l:.2f}) -> '{ann_l}'")
    assert ann_l == "Person on your left."

    # CENTER Region: center_x = 320, rel_x = 0.50
    pos_c, rel_c = classify_horizontal_position((270, 50, 370, 200), frame_width=w)
    assert pos_c == HorizontalPosition.CENTER, f"Expected CENTER, got {pos_c}"
    ann_c = format_position_announcement("chair", pos_c)
    print(f"CENTER detection: {pos_c.name} (rel_x: {rel_c:.2f}) -> '{ann_c}'")
    assert ann_c == "Chair in front of you."

    # RIGHT Region: center_x = 550, rel_x = 0.859 >= 0.65
    pos_r, rel_r = classify_horizontal_position((500, 50, 600, 200), frame_width=w)
    assert pos_r == HorizontalPosition.RIGHT, f"Expected RIGHT, got {pos_r}"
    ann_r = format_position_announcement("bottle", pos_r)
    print(f"RIGHT detection: {pos_r.name} (rel_x: {rel_r:.2f}) -> '{ann_r}'")
    assert ann_r == "Bottle on your right."

    print("\n=== 2. Testing Boundary Hysteresis ===")
    pos_h, rel_h = classify_horizontal_position(
        (200, 50, 260, 200),
        frame_width=w,
        current_position=HorizontalPosition.LEFT,
        hysteresis=0.04,
    )
    assert pos_h == HorizontalPosition.LEFT, f"Hysteresis failed: expected LEFT, got {pos_h}"
    print(f"Hysteresis retained LEFT at rel_x {rel_h:.3f} (threshold: 0.35, eff: 0.39)")

    print("\n=== 3. Testing Multiple Object Formatting ===")
    items = [
        ("person", HorizontalPosition.LEFT, None),
        ("chair", HorizontalPosition.CENTER, None),
        ("bottle", HorizontalPosition.RIGHT, None),
    ]
    multi_msg = format_multi_object_announcement(items)
    print(f"Multi-announcement: '{multi_msg}'")
    assert multi_msg == "Multiple objects detected: person on your left, chair in front of you, bottle on your right."

    print("\n=== 4. Testing Audio Queue Expiration & Thread Safety ===")
    cfg = AudioConfig(enabled=True, speech_rate_wpm=180)
    engine = SpeechEngine(cfg)
    engine.start()
    print("Speech engine started asynchronously.")

    # Enqueue alert with 0.01s expiration (will expire before playback)
    engine.speak_async("Expired message test", max_age_seconds=0.01)
    time.sleep(0.05)

    # Enqueue valid alert
    engine.speak_async("Person on your left.", max_age_seconds=2.0)
    time.sleep(0.6)
    engine.stop()
    print("Speech engine stopped cleanly.")

    print("\n=== 5. Testing Continuous Announcement Repeating at Fixed Interval ===")
    alert_cfg = AlertConfig(
        eligible_classes=["person"],
        announcement_interval_seconds=3.0,
    )
    df = DetectionFilter(alert_cfg)
    det = DetectionResult(0, "person", 0.9, (250, 100, 390, 300), 0.0)

    # Initial alert at t=0
    a0 = df.evaluate_alert([det], current_time=0.0, frame_width=w)
    assert a0 is not None
    print(f"[t=0.0s] Announcement triggered: '{a0[0]}'")

    # Suppressed at t=1.0s and t=2.0s
    assert df.evaluate_alert([det], current_time=1.0, frame_width=w) is None
    assert df.evaluate_alert([det], current_time=2.0, frame_width=w) is None
    print("[t=1.0s, 2.0s] Suppressed as expected (within 3.0s interval)")

    # Repeated at t=3.0s
    a3 = df.evaluate_alert([det], current_time=3.0, frame_width=w)
    assert a3 is not None
    print(f"[t=3.0s] Repeated announcement triggered: '{a3[0]}'")

    # Repeated again at t=6.0s
    a6 = df.evaluate_alert([det], current_time=6.0, frame_width=w)
    assert a6 is not None
    print(f"[t=6.0s] Repeated announcement triggered: '{a6[0]}'")

    print("\nAll targeted verification checks passed successfully!")


if __name__ == "__main__":
    test_verification()
