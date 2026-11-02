"""Backward compatibility bridge for audio module."""

from audio.speech import SpeechEngine, WindowsTTS

__all__ = ["SpeechEngine", "WindowsTTS"]
