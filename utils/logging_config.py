"""Thread-safe logging configuration with console output and GUI event dispatch."""

from __future__ import annotations

import logging
import queue
import sys
from typing import Optional

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_TIME_FORMAT = "%H:%M:%S"

# Global thread-safe queue for GUI consumption
gui_event_queue: queue.Queue[str] = queue.Queue(maxsize=200)


class EventLogHandler(logging.Handler):
    """Logging handler that pushes formatted messages into a thread-safe queue for UI display."""

    def __init__(self, target_queue: queue.Queue[str], level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.target_queue = target_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Drop older messages if queue is full
            try:
                self.target_queue.put_nowait(msg)
            except queue.Full:
                try:
                    self.target_queue.get_nowait()
                    self.target_queue.put_nowait(msg)
                except Exception:
                    pass
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO, ui_queue: Optional[queue.Queue[str]] = None) -> logging.Logger:
    """Configures root and application loggers."""
    root_logger = logging.getLogger("naviglass")
    root_logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not root_logger.handlers:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_TIME_FORMAT)

        # 1. Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. UI Event handler
        q = ui_queue if ui_queue is not None else gui_event_queue
        ui_handler = EventLogHandler(target_queue=q, level=level)
        ui_handler.setFormatter(formatter)
        root_logger.addHandler(ui_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Returns namespaced application logger."""
    return logging.getLogger(f"naviglass.{name}")
