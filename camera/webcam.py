"""Camera capture subsystem supporting OpenCV (DirectShow / V4L2) and Raspberry Pi Picamera2.

Provides a continuous-grabber architecture: a dedicated background thread
reads frames as fast as the camera delivers them, keeping only the latest
frame in a single-slot buffer. Consumer threads call ``read_latest_frame()``
and always get the most recent image without blocking on I/O.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Optional, Tuple
import cv2
import numpy as np

from config.settings import CameraConfig
from utils.logging_config import get_logger

logger = get_logger("camera")


class CameraError(Exception):
    """Raised when the camera fails to initialize, stream, or capture frames."""


class PiCamera2Adapter:
    """Hardware DMA capture adapter for Raspberry Pi CSI camera modules via Picamera2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as e:
            raise CameraError(
                "picamera2 is not installed. Install via: sudo apt install python3-picamera2"
            ) from e

        self.width = width
        self.height = height
        self.fps = fps
        self._picam2 = Picamera2()
        self._is_opened = False
        self._open()

    def _open(self) -> None:
        try:
            configured = False
            for fmt in ("BGR888", "R8", "XBGR8888"):
                try:
                    config = self._picam2.create_video_configuration(
                        main={"size": (self.width, self.height), "format": fmt},
                        controls={"FrameRate": self.fps},
                    )
                    self._picam2.configure(config)
                    logger.info("Picamera2 configured successfully with format: %s", fmt)
                    configured = True
                    break
                except Exception as fmt_err:
                    logger.debug("Picamera2 format %s unsupported: %s", fmt, fmt_err)

            if not configured:
                logger.warning("Picamera2 standard formats failed; trying default sensor format...")
                config = self._picam2.create_video_configuration(
                    main={"size": (self.width, self.height)},
                    controls={"FrameRate": self.fps},
                )
                self._picam2.configure(config)

            self._picam2.start()
            self._is_opened = True
        except Exception as e:
            self._is_opened = False
            raise CameraError(f"Failed to start Picamera2 device: {e}") from e

    def isOpened(self) -> bool:
        return self._is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_opened:
            return False, None
        try:
            frame = self._picam2.capture_array()
            if frame is None:
                return False, None
            return True, frame
        except Exception as e:
            logger.error("Picamera2 capture error: %s", e)
            return False, None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def release(self) -> None:
        self._is_opened = False
        try:
            self._picam2.stop()
        except Exception:
            pass
        try:
            self._picam2.close()
        except Exception:
            pass


class Webcam:
    """Encapsulates OpenCV VideoCapture / Picamera2 with a continuous grabber thread.

    The grabber thread runs ``cap.read()`` in a tight loop so that:
    * The OS/driver camera buffer is always drained (no stale frames).
    * ``read_latest_frame()`` returns in microseconds, never blocking
      on I/O — critical for keeping inference workers responsive.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: Optional[Any] = None
        self._lock = threading.Lock()
        self._is_running = False
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0

        # Continuous-grabber state
        self._grabber_thread: Optional[threading.Thread] = None
        self._stop_grabber = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_id: int = 0          # Monotonic counter
        self._grab_fps: float = 0.0      # Measured grabber FPS
        self._is_eof: bool = False       # True if media file source ended

        # Diagnostics & Recovery telemetry
        self._successful_reads: int = 0
        self._failed_reads: int = 0
        self._consecutive_failures: int = 0
        self._latest_capture_timestamp: float = 0.0
        self._recovery_attempts: int = 0

    @property
    def is_running(self) -> bool:
        return self._is_running and self._cap is not None and self._cap.isOpened()

    @property
    def grabber_alive(self) -> bool:
        """Returns True if the background grabber thread is actively running."""
        return self._grabber_thread is not None and self._grabber_thread.is_alive()

    @property
    def successful_reads(self) -> int:
        """Total number of successful frame reads since start."""
        return self._successful_reads

    @property
    def failed_reads(self) -> int:
        """Total number of failed frame read attempts since start."""
        return self._failed_reads

    @property
    def latest_capture_timestamp(self) -> float:
        """Unix timestamp of the most recent successfully captured frame."""
        return self._latest_capture_timestamp

    @property
    def dimensions(self) -> Tuple[int, int, float]:
        return (self._actual_width, self._actual_height, self._actual_fps)

    @property
    def grab_fps(self) -> float:
        """Measured frame-grab rate of the background grabber thread."""
        return self._grab_fps

    @property
    def frame_id(self) -> int:
        """Monotonically increasing frame counter (useful for skip logic)."""
        return self._frame_id

    @staticmethod
    def _normalize_frame(
        frame: Optional[np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """Ensures captured frame is converted to standard 3-channel BGR format.

        Handles monochrome 10-bit / 8-bit sensors (such as OV9281 on Raspberry Pi),
        single-channel arrays, and 4-channel BGRA/XBGR images. Optionally resizes to target_size (w, h).
        """
        if frame is None:
            return None
        # Convert 16-bit monochrome (e.g. 10-bit mono from OV9281 R10_CSI2P) to 8-bit
        if frame.dtype == np.uint16:
            frame = (frame >> 2).astype(np.uint8)
        # Convert 2D grayscale to 3-channel BGR
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Convert 3D 1-channel to 3-channel BGR
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Convert 4-channel BGRA/XBGR to 3-channel BGR
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Scale to target dimensions if requested and different
        if target_size and len(target_size) == 2:
            tw, th = target_size
            if tw > 0 and th > 0:
                h, w = frame.shape[:2]
                if (w, h) != (tw, th):
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

        return frame

    def _open_capture_device(self, source: Any) -> Any:
        """Opens video capture device using Picamera2 or platform-appropriate OpenCV backend."""
        backend = getattr(self.config, "backend", "auto").lower()
        is_picam_req = backend == "picamera2" or str(source).lower() in ("picamera2", "csi", "rpi")

        # Try Picamera2 when explicitly requested or on Linux in auto mode without explicit file source
        if is_picam_req or (backend == "auto" and sys.platform.startswith("linux") and not self.config.source):
            try:
                logger.info(
                    "Attempting to initialize Picamera2 device (%dx%d @ %d FPS)...",
                    self.config.width,
                    self.config.height,
                    self.config.fps,
                )
                adapter = PiCamera2Adapter(
                    width=self.config.width,
                    height=self.config.height,
                    fps=self.config.fps,
                )
                if adapter.isOpened():
                    logger.info("Picamera2 device initialized successfully.")
                    return adapter
            except Exception as picam_err:
                if is_picam_req:
                    raise CameraError(f"Requested Picamera2 backend failed to initialize: {picam_err}") from picam_err
                logger.info("Picamera2 not available or no CSI camera detected (%s); falling back to OpenCV...", picam_err)

        # Standard OpenCV VideoCapture backend selection
        cap = None
        if isinstance(source, int):
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    logger.info("DirectShow open failed, trying default backend...")
                    cap.release()
                    cap = cv2.VideoCapture(source)
            elif sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                if not cap.isOpened():
                    logger.info("V4L2 open failed, trying default backend...")
                    cap.release()
                    cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)
        else:
            # Video file or image source
            cap = cv2.VideoCapture(str(source))

        return cap

    def start(self) -> None:
        """Initializes the camera device and launches the grabber thread.

        Raises:
            CameraError: If device cannot be accessed or permission is denied.
        """
        with self._lock:
            if self.is_running:
                logger.warning("Camera is already active.")
                return

            source = self.config.source if self.config.source else self.config.index
            logger.info("Starting camera device (source: %s)...", source)

            cap = self._open_capture_device(source)

            if cap is None or not cap.isOpened():
                raise CameraError(
                    f"Failed to open video source: {source}. "
                    "Ensure webcam or Raspberry Pi camera is connected, permissions are granted, "
                    "and no other application is using it."
                )

            # Set requested resolution & FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Reduce internal OpenCV buffer to 1 to minimise stale-frame lag
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify by capturing a single test frame
            try:
                ret, raw_test_frame = cap.read()
            except Exception as read_err:
                cap.release()
                raise CameraError(f"Exception during test frame capture from source {source}: {read_err}")

            test_frame = self._normalize_frame(raw_test_frame, (self.config.width, self.config.height))

            if not ret or test_frame is None:
                cap.release()
                raise CameraError(
                    f"Camera opened at source {source} but failed to capture a test frame. "
                    "Ensure device is not occupied by another application."
                )

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or test_frame.shape[1]
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or test_frame.shape[0]
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.config.fps)

            self._actual_width = actual_w
            self._actual_height = actual_h
            self._actual_fps = actual_fps
            self._cap = cap
            self._is_running = True
            self._is_eof = False
            self._frame_id = 0
            self._successful_reads = 1
            self._failed_reads = 0
            self._consecutive_failures = 0
            self._latest_capture_timestamp = time.time()

            # Store the test frame as the initial latest frame
            with self._frame_lock:
                self._latest_frame = test_frame

            logger.info(
                "Camera started successfully: %dx%d @ %.1f FPS",
                self._actual_width,
                self._actual_height,
                self._actual_fps,
            )

        # Launch continuous grabber thread
        self._stop_grabber.clear()
        self._grabber_thread = threading.Thread(
            target=self._grabber_loop,
            name="NaviGlassCameraGrabber",
            daemon=True,
        )
        self._grabber_thread.start()
        logger.info("Camera grabber thread started.")

    def _attempt_camera_recovery(self) -> bool:
        """Controlled recovery when camera read fails repeatedly."""
        with self._lock:
            self._recovery_attempts += 1
            logger.warning(
                "Triggering controlled camera recovery (attempt %d, source: %s)...",
                self._recovery_attempts,
                self.config.source or self.config.index,
            )
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception as e:
                logger.warning("Error releasing cap during recovery: %s", e)
            self._cap = None
            time.sleep(0.3)

            source = self.config.source if self.config.source else self.config.index
            try:
                cap = self._open_capture_device(source)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                    cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, raw_test_frame = cap.read()
                    test_frame = self._normalize_frame(raw_test_frame, (self.config.width, self.config.height))
                    if ret and test_frame is not None:
                        self._cap = cap
                        self._consecutive_failures = 0
                        with self._frame_lock:
                            self._latest_frame = test_frame
                            self._latest_capture_timestamp = time.time()
                        logger.info("Camera recovery successful.")
                        return True
                    else:
                        cap.release()
            except Exception as e:
                logger.error("Camera recovery attempt failed: %s", e)

            logger.error("Camera recovery failed. Will retry.")
            return False

    def _grabber_loop(self) -> None:
        """Continuously reads frames from the camera, keeping only the latest."""
        fps_counter = 0
        fps_t0 = time.perf_counter()

        while not self._stop_grabber.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                if not self.config.source and not self._stop_grabber.is_set():
                    self._attempt_camera_recovery()
                time.sleep(0.02)
                continue

            try:
                ret, raw_frame = cap.read()
                frame = self._normalize_frame(raw_frame, (self.config.width, self.config.height))
            except Exception as e:
                self._failed_reads += 1
                self._consecutive_failures += 1
                logger.error("Camera grabber read error: %s", e)
                if self._consecutive_failures >= 30 and not self.config.source:
                    self._attempt_camera_recovery()
                time.sleep(0.01)
                continue

            if not ret or frame is None:
                self._failed_reads += 1
                self._consecutive_failures += 1
                if self.config.source:
                    with self._frame_lock:
                        self._is_eof = True
                    break

                if self._consecutive_failures >= 30 and not self.config.source:
                    logger.warning(
                        "Camera read failed for %d consecutive frames. Initiating recovery...",
                        self._consecutive_failures,
                    )
                    with self._frame_lock:
                        if self._consecutive_failures >= 60:
                            # Do not silently display an old frame indefinitely during hardware stall
                            self._latest_frame = None
                    self._attempt_camera_recovery()

                time.sleep(0.005)
                continue

            # Successful capture
            now = time.time()
            self._successful_reads += 1
            self._consecutive_failures = 0
            self._latest_capture_timestamp = now

            with self._frame_lock:
                self._latest_frame = frame
                self._frame_id += 1

            fps_counter += 1
            elapsed = time.perf_counter() - fps_t0
            if elapsed >= 1.0:
                self._grab_fps = fps_counter / elapsed
                fps_counter = 0
                fps_t0 = time.perf_counter()

        logger.info("Camera grabber thread stopped.")

    def read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Returns the most recent frame captured by the grabber thread.

        This call is non-blocking and returns in microseconds.

        Returns:
            Tuple of (success, frame_bgr, frame_id).
        """
        with self._frame_lock:
            if not self._is_running or self._is_eof or self._latest_frame is None:
                return False, None, self._frame_id
            return True, self._latest_frame, self._frame_id

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Legacy API: reads the latest frame (backward-compatible).

        Returns:
            Tuple of (success, frame_bgr).
        """
        ok, frame, _ = self.read_latest_frame()
        return ok, frame

    def show_preview(self, frame: np.ndarray, title: str = "NaviGlass Camera Preview", fps: float = 0.0) -> bool:
        """Displays preview window if graphical display is available.

        Returns False if user pressed 'q' or ESC to request exit, True otherwise.
        Safely ignores errors in headless environments without display.
        """
        try:
            disp = frame.copy()
            if fps > 0:
                cv2.putText(
                    disp,
                    f"FPS: {fps:.1f}",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(title, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return False
            return True
        except cv2.error as e:
            logger.debug("show_preview skipped (no display or headless cv2): %s", e)
            return True
        except Exception as e:
            logger.debug("show_preview unexpected error: %s", e)
            return True

    def stop(self) -> None:
        """Releases camera hardware resources safely."""
        # Stop grabber thread first
        self._stop_grabber.set()
        if self._grabber_thread and self._grabber_thread.is_alive():
            self._grabber_thread.join(timeout=2.0)
        self._grabber_thread = None

        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.warning("Error releasing camera: %s", e)
                finally:
                    self._cap = None
                    self._is_running = False
                    with self._frame_lock:
                        self._latest_frame = None
                    logger.info("Camera stopped and hardware resources released.")


# Alias for backward compatibility
OpenCVCamera = Webcam
