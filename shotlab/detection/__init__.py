"""Offline boundary proposals. No Qt, Project, or ProjectStore dependency."""

from .core import Boundary, DetectionResult, Detector, FrameSource, run

__all__ = ["Boundary", "DetectionResult", "Detector", "FrameSource", "run"]
