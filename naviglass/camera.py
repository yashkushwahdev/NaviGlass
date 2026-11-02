"""Backward compatibility bridge for camera module."""

from camera.webcam import CameraError, OpenCVCamera, Webcam

__all__ = ["CameraError", "OpenCVCamera", "Webcam"]
