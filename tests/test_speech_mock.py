"""Unit tests for speech queue behavior, expiration, duplicate suppression, and mute controls."""

import time
import pytest
from unittest.mock import MagicMock, patch
from config.settings import AudioConfig
from audio.speech import SpeechEngine


def test_speech_disabled():
    cfg = AudioConfig(enabled=False)
    engine = SpeechEngine(cfg)

    # Should immediately return False without queuing
    assert engine.speak_async("Hello") is False


def test_speech_muted():
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)
    engine.set_muted(True)

    assert engine.is_muted is True
    assert engine.speak_async("Alert") is False


def test_speech_queue_drop_when_full():
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)
    engine._is_available = True

    # Fill queue (capacity is 3) with distinct messages
    assert engine.speak_async("Alert 1") is True
    assert engine.speak_async("Alert 2") is True
    assert engine.speak_async("Alert 3") is True

    # 4th distinct item should be dropped gracefully to prevent backlog
    assert engine.speak_async("Alert 4") is False


def test_speech_duplicate_suppression():
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)
    engine._is_available = True

    # First enqueue succeeds
    assert engine.speak_async("Person on your left.") is True

    # Immediate duplicate request should be suppressed
    assert engine.speak_async("Person on your left.") is False

    # Distinct request succeeds
    assert engine.speak_async("Chair in front of you.") is True


def test_speech_mute_clears_queue():
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)
    engine._is_available = True

    engine.speak_async("Message 1")
    engine.speak_async("Message 2")
    assert not engine._queue.empty()

    # Muting should purge queue
    engine.set_muted(True)
    assert engine._queue.empty()


def test_speech_fallback_to_terminal_when_unavailable():
    cfg = AudioConfig(enabled=True, fallback_to_terminal=True)
    engine = SpeechEngine(cfg)
    engine._is_available = False

    # Should return True (fallback handled)
    assert engine.speak_async("Test announcement") is True


def test_worker_sequential_playback_mock():
    """Verifies worker thread processes consecutive items and remains alive."""
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)

    spoken_items = []

    mock_pyttsx3_engine = MagicMock()
    def fake_say(text):
        spoken_items.append(text)
    mock_pyttsx3_engine.say.side_effect = fake_say

    with patch("pyttsx3.init", return_value=mock_pyttsx3_engine):
        engine.start()
        assert engine._worker_thread.is_alive()

        assert engine.speak_async("First announcement") is True
        engine.wait_until_done(timeout=2.0)

        assert engine.speak_async("Second announcement") is True
        engine.wait_until_done(timeout=2.0)

        assert engine.speak_async("Third announcement") is True
        engine.wait_until_done(timeout=2.0)

        engine.stop()
        assert not engine._worker_thread.is_alive()

    assert spoken_items == ["First announcement", "Second announcement", "Third announcement"]


def test_worker_linux_espeak_playback_mock():
    """Verifies speech playback on Linux with native espeak-ng/espeak via subprocess."""
    import sys
    cfg = AudioConfig(enabled=True, speech_rate_wpm=180, volume=0.8)
    engine = SpeechEngine(cfg)

    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        with patch.object(sys, "platform", "linux"):
            engine.start()
            assert engine._backend_type == "espeak"
            assert engine.is_available is True

            with patch("subprocess.run") as mock_sub:
                assert engine.speak_async("Caution, chair in front of you.") is True
                engine.wait_until_done(timeout=2.0)
                mock_sub.assert_called_once()
                args, kwargs = mock_sub.call_args
                cmd = args[0]
                assert cmd[0] == "/usr/bin/espeak-ng"
                assert "180" in cmd
                assert "Caution, chair in front of you." in cmd

            engine.stop()
            assert not engine._worker_thread.is_alive()

