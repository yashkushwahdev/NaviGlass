"""Backward compatibility bridge for detection module."""

from detection.detector import DetectionResult, DetectorError, ObjectDetector
from detection.nearest_object import NearestObjectManager, WarningDistanceLevel

__all__ = ["DetectionResult", "DetectorError", "ObjectDetector", "NearestObjectManager", "WarningDistanceLevel"]

