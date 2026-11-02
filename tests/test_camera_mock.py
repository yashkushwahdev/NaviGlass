"""Unit tests for camera initialization, frame capture, and error handling with mocks."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from camera.webcam import CameraError, Webcam
from config.settings import CameraConfig


def test_camera_missing_raises_camera_error():
    cfg = CameraConfig(index=99)
    webcam = Webcam(cfg)

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        with pytest.raises(CameraError) as exc_info:
            webcam.start()

        assert "Failed to open video source" in str(exc_info.value)
        assert webcam.is_running is False


def test_camera_successful_start_and_read():
    cfg = CameraConfig(index=0, width=640, height=480, fps=30)
    webcam = Webcam(cfg)

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == 3 else (480.0 if prop == 4 else 30.0)
        mock_cap_cls.return_value = mock_cap

        webcam.start()
        assert webcam.is_running is True

        success, frame = webcam.read_frame()
        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

        webcam.stop()
        assert webcam.is_running is False
        mock_cap.release.assert_called_once()


def test_camera_normalize_monochrome_sensor():
    """Verifies that 10-bit uint16 and 8-bit uint8 monochrome frames (e.g. OV9281) are normalized to 3-channel BGR."""
    # 10-bit uint16 mono
    mono_10bit = np.full((480, 640), 1023, dtype=np.uint16)
    bgr_from_10bit = Webcam._normalize_frame(mono_10bit)
    assert bgr_from_10bit is not None
    assert bgr_from_10bit.shape == (480, 640, 3)
    assert bgr_from_10bit.dtype == np.uint8
    assert bgr_from_10bit[0, 0, 0] == 255

    # 8-bit uint8 mono
    mono_8bit = np.full((480, 640), 128, dtype=np.uint8)
    bgr_from_8bit = Webcam._normalize_frame(mono_8bit)
    assert bgr_from_8bit is not None
    assert bgr_from_8bit.shape == (480, 640, 3)
    assert bgr_from_8bit.dtype == np.uint8

    # 4-channel BGRA
    bgra = np.zeros((480, 640, 4), dtype=np.uint8)
    bgr_from_bgra = Webcam._normalize_frame(bgra)
    assert bgr_from_bgra is not None
    assert bgr_from_bgra.shape == (480, 640, 3)


def test_camera_normalize_ov9281_native_scaling():
    """Verifies that OV9281 1280x800 monochrome 10-bit is scaled to target 640x480 3-channel BGR."""
    native_frame = np.full((800, 1280), 512, dtype=np.uint16)
    out_frame = Webcam._normalize_frame(native_frame, target_size=(640, 480))
    assert out_frame is not None
    assert out_frame.shape == (480, 640, 3)
    assert out_frame.dtype == np.uint8
    assert out_frame[0, 0, 0] == 128


def test_camera_show_preview_headless_safe():
    """Verifies show_preview does not raise exceptions even when cv2 fails in headless mode."""
    cfg = CameraConfig(index=0)
    webcam = Webcam(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.imshow", side_effect=Exception("No display")):
        assert webcam.show_preview(frame) is True
