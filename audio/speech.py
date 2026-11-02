"""Platform-adaptive asynchronous text-to-speech engine supporting Linux (espeak-ng / espeak) and Windows (SAPI5 / pyttsx3)."""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional, Tuple

from config.settings import AudioConfig
from utils.logging_config import get_logger

logger = get_logger("audio")


class SpeechEngine:
    """Asynchronous, non-blocking text-to-speech engine with worker queue, message expiration, and mute controls."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        # Queue contains tuples of (text, enqueue_timestamp, max_age_seconds)
        self._queue: queue.Queue[Tuple[str, float, float]] = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._is_available = False
        self._is_muted = False
        self._is_enabled = config.enabled
        self._is_speaking = False
        self._last_spoken_text = ""
        self._last_spoken_time = 0.0
        self._last_enqueued_text = ""
        self._last_enqueued_time = 0.0
        self._backend_type = "terminal"  # "espeak", "pyttsx3", "terminal"
        self._espeak_cmd: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @property
    def is_speaking(self) -> bool:
        """Returns True if the TTS worker is actively executing speech playback."""
        return self._is_speaking

    def set_muted(self, muted: bool) -> None:
        """Sets mute state. When muted, flushes any pending queued alerts."""
        self._is_muted = muted
        if muted:
            self.clear_queue()
        logger.info("Speech muted set to: %s", muted)

    def set_enabled(self, enabled: bool) -> None:
        """Sets enabled state. When disabled, flushes any pending queued alerts."""
        self._is_enabled = enabled
        if not enabled:
            self.clear_queue()
        logger.info("Speech enabled set to: %s", enabled)

    def clear_queue(self) -> None:
        """Flushes all pending speech items from the queue."""
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                cleared += 1
            except (queue.Empty, ValueError):
                break
        if cleared > 0:
            logger.debug("Cleared %d pending alerts from speech queue.", cleared)

    def start(self) -> None:
        """Initializes the platform-appropriate TTS engine and worker thread."""
        espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")

        # On Linux / Raspberry Pi, prefer native subprocess espeak-ng/espeak for crash-free GIL-less audio
        if sys.platform.startswith("linux") and espeak_bin:
            self._backend_type = "espeak"
            self._espeak_cmd = espeak_bin
            self._is_available = True
            logger.info("Linux native speech engine (%s) initialized successfully.", espeak_bin)
        else:
            # On Windows or fallback: use pyttsx3 (SAPI5)
            try:
                import pyttsx3  # type: ignore
                test_eng = pyttsx3.init()
                test_eng.stop()
                del test_eng
                self._backend_type = "pyttsx3"
                self._is_available = True
                eng_name = "Windows SAPI5" if sys.platform.startswith("win") else "pyttsx3"
                logger.info("%s TTS engine initialized successfully.", eng_name)
            except Exception as e:
                if espeak_bin:
                    self._backend_type = "espeak"
                    self._espeak_cmd = espeak_bin
                    self._is_available = True
                    logger.info("Fell back to Linux native speech engine (%s).", espeak_bin)
                else:
                    self._backend_type = "terminal"
                    self._is_available = False
                    logger.warning("Could not initialize text-to-speech backend: %s", e)
                    if self.config.fallback_to_terminal:
                        logger.info("Falling back to terminal-only announcements.")

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="NaviGlassSpeechWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Speech worker thread started.")

    def speak_async(
        self,
        text: str,
        max_age_seconds: float = 4.0,
        priority: bool = False,
    ) -> bool:
        """Enqueues text for asynchronous speech output.

        If priority=True (e.g. urgent proximity alert under 1m), pending lower-priority
        alerts in the queue are flushed to avoid delaying immediate critical warnings.
        """
        clean_text = text.strip()
        if not clean_text:
            return False

        if not self._is_enabled or self._is_muted:
            logger.debug("Speech suppressed (enabled=%s, muted=%s): '%s'", self._is_enabled, self._is_muted, clean_text)
            return False

        now = time.time()

        if clean_text == self._last_enqueued_text and (now - self._last_enqueued_time) < 0.25:
            logger.debug("Speech suppressed (rapid duplicate burst < 0.25s): '%s'", clean_text)
            return False

        with self._queue.mutex:
            for item in list(self._queue.queue):
                if item[0] == clean_text:
                    logger.debug("Speech suppressed (already waiting in queue): '%s'", clean_text)
                    return False

        # If urgent priority, flush older pending alerts to make room immediately
        if priority:
            self.clear_queue()
            max_age_seconds = min(max_age_seconds, 2.5)

        if not self._is_available:
            logger.info('[AUDIO TERMINAL FALLBACK%s]: "%s"', " (PRIORITY)" if priority else "", clean_text)
            self._last_spoken_text = clean_text
            self._last_spoken_time = now
            self._last_enqueued_text = clean_text
            self._last_enqueued_time = now
            return True

        item = (clean_text, now, max_age_seconds)
        try:
            self._queue.put_nowait(item)
            self._last_enqueued_text = clean_text
            self._last_enqueued_time = now
            logger.info(
                "Speech enqueued%s: '%s' (max_age=%.1fs, queue_len=%d)",
                " [PRIORITY]" if priority else "",
                clean_text,
                max_age_seconds,
                self._queue.qsize(),
            )
            return True
        except queue.Full:
            logger.warning("Speech queue full. Dropped stale alert: '%s'", clean_text)
            return False

    def wait_until_done(self, timeout: float = 8.0) -> None:
        """Waits until pending speech utterances in the queue are completed."""
        start_t = time.time()
        while (not self._queue.empty() or self._is_speaking) and (time.time() - start_t) < timeout:
            time.sleep(0.05)

    def _worker_loop(self) -> None:
        """Dedicated background speech loop ensuring thread safety on Linux and COM threading model on Windows."""
        if sys.platform.startswith("win"):
            try:
                import pythoncom  # type: ignore
                pythoncom.CoInitialize()
                logger.debug("COM initialized on speech worker thread.")
            except Exception as e:
                logger.debug("COM initialization note: %s", e)

        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if self._stop_event.is_set():
                    break

                text, enqueue_time, max_age = item
                now = time.time()
                latency = now - enqueue_time
                logger.info("Speech dequeued: '%s' (latency: %.2fs)", text, latency)

                if latency > max_age:
                    logger.info("Speech dropped (expired after %.2fs > %.2fs): '%s'", latency, max_age, text)
                    self._queue.task_done()
                    continue

                if not self._is_muted and self._is_enabled:
                    try:
                        self._is_speaking = True
                        self._last_spoken_text = text
                        t_speak_start = time.time()
                        logger.info("TTS starting playback: '%s'", text)

                        if self._backend_type == "espeak" and self._espeak_cmd:
                            vol = max(10, min(200, int(self.config.volume * 100)))
                            cmd = [
                                self._espeak_cmd,
                                "-s", str(self.config.speech_rate_wpm),
                                "-a", str(vol),
                                text,
                            ]
                            subprocess.run(
                                cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=6.0,
                                check=False,
                            )
                        elif self._backend_type == "pyttsx3":
                            import pyttsx3  # type: ignore
                            engine = pyttsx3.init()
                            try:
                                engine.setProperty("rate", self.config.speech_rate_wpm)
                                engine.setProperty("volume", self.config.volume)
                                engine.say(text)
                                engine.runAndWait()
                            finally:
                                try:
                                    engine.stop()
                                except Exception:
                                    pass
                                del engine
                        else:
                            if self.config.fallback_to_terminal:
                                logger.info('[AUDIO TERMINAL FALLBACK]: "%s"', text)

                        self._last_spoken_time = time.time()
                        logger.info(
                            "TTS completed playback: '%s' (duration: %.2fs)",
                            text,
                            self._last_spoken_time - t_speak_start,
                        )
                    except Exception as e:
                        logger.error("TTS speaking error for '%s': %s", text, e)
                        if self.config.fallback_to_terminal:
                            logger.info('[AUDIO TERMINAL FALLBACK]: "%s"', text)
                    finally:
                        self._is_speaking = False

                self._queue.task_done()
        except Exception as outer_e:
            logger.error("Unexpected fatal exception in speech worker loop: %s", outer_e)
        finally:
            self._is_speaking = False
            if sys.platform.startswith("win"):
                try:
                    import pythoncom  # type: ignore
                    pythoncom.CoUninitialize()
                    logger.debug("COM uninitialized on speech worker thread.")
                except Exception:
                    pass
            logger.info("Speech worker loop exited.")

    def stop(self) -> None:
        """Shuts down the speech worker cleanly."""
        self._stop_event.set()
        self.clear_queue()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("Speech engine stopped cleanly.")


# Aliases for cross-platform and backward compatibility
WindowsTTS = SpeechEngine
LinuxTTS = SpeechEngine
