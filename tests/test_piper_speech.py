"""Automated tests for Piper TTS integration, en_US-amy-medium voice model, and platform fallbacks.

Tests cover:
1. Piper module import
2. Amy voice model loading and config verification
3. WAV generation via Piper Python API
4. Temporary WAV file creation, playback, and cleanup
5. SpeechEngine integration with Piper as primary Windows backend
6. Sequential consecutive announcements through SpeechEngine
7. Priority announcements purging lower-priority backlog
8. Mute controls and queue clearing
9. Graceful fallback to secondary TTS when Piper fails
10. Linux / Raspberry Pi fallback path verification
"""

from __future__ import annotations

import sys
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from config.settings import AudioConfig
from audio.speech import SpeechEngine


MODEL_PATH = Path("models/voices/en_US-amy-medium.onnx")
CONFIG_PATH = Path("models/voices/en_US-amy-medium.onnx.json")


# ---------------------------------------------------------------------------
# Test 1: Piper module import
# ---------------------------------------------------------------------------
def test_piper_import():
    """Verify piper and PiperVoice can be imported."""
    import piper
    from piper.voice import PiperVoice

    assert piper is not None
    assert callable(PiperVoice.load)


# ---------------------------------------------------------------------------
# Test 2: Amy voice model loading
# ---------------------------------------------------------------------------
def test_piper_amy_model_loading():
    """Verify loading the official en_US-amy-medium ONNX model and config."""
    assert MODEL_PATH.exists(), f"Voice model missing at {MODEL_PATH}"
    assert CONFIG_PATH.exists(), f"Voice config missing at {CONFIG_PATH}"

    from piper.voice import PiperVoice

    voice = PiperVoice.load(str(MODEL_PATH), config_path=str(CONFIG_PATH), use_cuda=False)
    assert voice is not None
    assert voice.config.sample_rate == 22050


# ---------------------------------------------------------------------------
# Test 3: WAV generation via Piper
# ---------------------------------------------------------------------------
def test_piper_wav_generation():
    """Verify synthesizing text directly to a valid WAV file."""
    from piper.voice import PiperVoice

    voice = PiperVoice.load(str(MODEL_PATH), config_path=str(CONFIG_PATH), use_cuda=False)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_wav = Path(f.name)

    try:
        with wave.open(str(temp_wav), "wb") as wav_file:
            voice.synthesize_wav("Caution, chair in front of you.", wav_file)

        assert temp_wav.exists()
        assert temp_wav.stat().st_size > 10_000

        with wave.open(str(temp_wav), "rb") as r:
            assert r.getnchannels() == 1
            assert r.getframerate() == 22050
            assert r.getnframes() > 0
    finally:
        if temp_wav.exists():
            temp_wav.unlink()


# ---------------------------------------------------------------------------
# Test 4: Temporary WAV file cleanup
# ---------------------------------------------------------------------------
def test_piper_temporary_file_cleanup():
    """Verify SpeechEngine._play_piper always removes temporary WAV files after playback."""
    cfg = AudioConfig(
        enabled=True,
        voice_model=str(MODEL_PATH),
        voice_config=str(CONFIG_PATH),
    )
    engine = SpeechEngine(cfg)
    assert engine._init_piper() is True

    created_wavs = []
    orig_named_temp = tempfile.NamedTemporaryFile

    def tracking_temp(*args, **kwargs):
        res = orig_named_temp(*args, **kwargs)
        created_wavs.append(Path(res.name))
        return res

    with patch("tempfile.NamedTemporaryFile", side_effect=tracking_temp):
        with patch("winsound.PlaySound") as mock_play:
            engine._play_piper("Testing cleanup.")
            mock_play.assert_called_once()

    assert len(created_wavs) == 1
    # Verify the temporary WAV file has been removed
    assert not created_wavs[0].exists()


# ---------------------------------------------------------------------------
# Test 5: SpeechEngine with Piper as primary Windows backend
# ---------------------------------------------------------------------------
def test_speech_engine_piper_primary():
    """Verify SpeechEngine automatically selects Piper Amy on Windows."""
    cfg = AudioConfig(
        enabled=True,
        voice_model=str(MODEL_PATH),
        voice_config=str(CONFIG_PATH),
    )
    engine = SpeechEngine(cfg)

    with patch("winsound.PlaySound") as mock_play:
        engine.start()
        assert engine.is_available is True
        assert engine.backend_type == "piper"
        assert engine.piper_voice is not None

        spoken = engine.speak_async("Person on your left.")
        assert spoken is True
        engine.wait_until_done(timeout=3.0)
        mock_play.assert_called_once()

        engine.stop()
        assert not engine._worker_thread.is_alive()


# ---------------------------------------------------------------------------
# Test 6: Two consecutive announcements
# ---------------------------------------------------------------------------
def test_speech_engine_consecutive_announcements():
    """Verify worker thread processes consecutive announcements sequentially without stalling."""
    cfg = AudioConfig(
        enabled=True,
        voice_model=str(MODEL_PATH),
        voice_config=str(CONFIG_PATH),
    )
    engine = SpeechEngine(cfg)

    played_items = []
    def fake_play(path, flags):
        played_items.append(path)

    with patch("winsound.PlaySound", side_effect=fake_play):
        engine.start()

        assert engine.speak_async("First announcement.") is True
        assert engine.speak_async("Second announcement.") is True

        engine.wait_until_done(timeout=4.0)
        assert len(played_items) == 2

        engine.stop()


# ---------------------------------------------------------------------------
# Test 7: Priority announcements
# ---------------------------------------------------------------------------
def test_speech_engine_priority_announcement():
    """Verify urgent priority announcement flushes lower-priority backlog."""
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)
    engine._is_available = True
    engine._backend_type = "piper"

    # Enqueue standard announcements
    assert engine.speak_async("Standard notice 1", priority=False) is True
    assert engine.speak_async("Standard notice 2", priority=False) is True
    assert engine._queue.qsize() == 2

    # Enqueue urgent priority alert
    assert engine.speak_async("Warning! Obstacle ahead.", priority=True) is True

    # Previous items must have been flushed; only priority item remains
    assert engine._queue.qsize() == 1
    item = engine._queue.get_nowait()
    assert item[0] == "Warning! Obstacle ahead."


# ---------------------------------------------------------------------------
# Test 8: Mute and queue clearing
# ---------------------------------------------------------------------------
def test_speech_engine_mute_and_clear_queue():
    """Verify setting muted suppresses speech and flushes queue."""
    cfg = AudioConfig(enabled=True)
    engine = SpeechEngine(cfg)
    engine._is_available = True
    engine._backend_type = "piper"

    engine.speak_async("Test 1")
    assert not engine._queue.empty()

    engine.set_muted(True)
    assert engine.is_muted is True
    assert engine._queue.empty()

    # While muted, new speech requests are rejected
    assert engine.speak_async("Test 2") is False


# ---------------------------------------------------------------------------
# Test 9: Fallback when Piper fails
# ---------------------------------------------------------------------------
def test_speech_engine_fallback_when_piper_fails():
    """Verify that if Piper model is missing or fails, engine falls back to secondary TTS without crashing."""
    cfg = AudioConfig(
        enabled=True,
        voice_model="models/voices/nonexistent_voice.onnx",
        voice_config="models/voices/nonexistent_voice.onnx.json",
    )
    engine = SpeechEngine(cfg)

    mock_pyttsx3 = MagicMock()
    with patch("pyttsx3.init", return_value=mock_pyttsx3):
        engine.start()
        # Piper must fail and fall back to pyttsx3 or terminal
        assert engine.backend_type in ("pyttsx3", "terminal")
        assert engine.piper_voice is None

        engine.stop()


# ---------------------------------------------------------------------------
# Test 10: Linux / Raspberry Pi fallback path verification
# ---------------------------------------------------------------------------
def test_speech_engine_linux_fallback_path():
    """Verify that on Linux when Piper model is missing, it falls back to espeak-ng."""
    cfg = AudioConfig(
        enabled=True,
        voice_model="models/voices/nonexistent_voice.onnx",
    )
    engine = SpeechEngine(cfg)

    with patch("shutil.which", return_value="/usr/bin/espeak-ng"):
        with patch.object(sys, "platform", "linux"):
            engine.start()
            assert engine.backend_type == "espeak"
            assert engine.is_available is True
            assert engine.piper_voice is None

            engine.stop()

