"""Backward compatibility bridge for filtering module."""

from detection.filtering import AlertManager, DetectionFilter

__all__ = ["AlertManager", "DetectionFilter"]
