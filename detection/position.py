"""Relative horizontal object position classification and voice guidance formatting.

NOTE ON COORDINATE SYSTEM:
Position classifications ('on your left', 'in front of you', 'on your right')
strictly refer to the horizontal location of the detected object within the
camera image frame (monocular field of view). They do NOT infer the user's
physical body orientation, walking direction, or real-world compass coordinates.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple


class HorizontalPosition(str, Enum):
    """Relative horizontal position of an object in the camera image frame."""
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"

    @property
    def phrase(self) -> str:
        """Natural spoken English phrase describing the position in the camera view."""
        if self == HorizontalPosition.LEFT:
            return "on your left"
        elif self == HorizontalPosition.CENTER:
            return "in front of you"
        elif self == HorizontalPosition.RIGHT:
            return "on your right"
        return "in front of you"

    @property
    def label(self) -> str:
        """Short UI label."""
        if self == HorizontalPosition.LEFT:
            return "Left"
        elif self == HorizontalPosition.CENTER:
            return "Center"
        elif self == HorizontalPosition.RIGHT:
            return "Right"
        return "Center"


def classify_horizontal_position(
    box: Tuple[float, float, float, float],
    frame_width: int = 640,
    left_threshold: float = 0.35,
    right_threshold: float = 0.65,
    current_position: Optional[HorizontalPosition] = None,
    hysteresis: float = 0.04,
) -> Tuple[HorizontalPosition, float]:
    """Classifies an object bounding box horizontal position relative to the camera field of view.

    Coordinate Calculation:
        center_x = (x1 + x2) / 2
        relative_x = center_x / frame_width

    Threshold Classification:
        - relative_x < left_threshold (0.35): LEFT ("on your left")
        - left_threshold <= relative_x < right_threshold (0.65): CENTER ("in front of you")
        - relative_x >= right_threshold (0.65): RIGHT ("on your right")

    Hysteresis Band:
        Applies a +/- hysteresis margin (default 0.04) to prevent rapid speech
        oscillation when an object sits near a region boundary.

    Args:
        box: (x1, y1, x2, y2) in pixel coordinates.
        frame_width: Total pixel width of the camera frame (default: 640).
        left_threshold: Relative X threshold below which is LEFT (default: 0.35).
        right_threshold: Relative X threshold at or above which is RIGHT (default: 0.65).
        current_position: Previous classified position if tracking an object, for hysteresis.
        hysteresis: Buffer band to prevent oscillation near region boundaries.

    Returns:
        Tuple of (HorizontalPosition, relative_x ratio between 0.0 and 1.0).
    """
    x1, _, x2, _ = box
    center_x = (x1 + x2) / 2.0
    w = max(1.0, float(frame_width))
    relative_x = float(center_x / w)

    eff_left = left_threshold
    eff_right = right_threshold

    if current_position == HorizontalPosition.LEFT:
        eff_left += hysteresis
    elif current_position == HorizontalPosition.RIGHT:
        eff_right -= hysteresis
    elif current_position == HorizontalPosition.CENTER:
        eff_left -= hysteresis
        eff_right += hysteresis

    if relative_x < eff_left:
        pos = HorizontalPosition.LEFT
    elif relative_x >= eff_right:
        pos = HorizontalPosition.RIGHT
    else:
        pos = HorizontalPosition.CENTER

    return pos, relative_x


def format_position_announcement(
    class_name: str,
    position: HorizontalPosition,
    distance: Optional[float] = None,
) -> str:
    """Builds a concise, natural spoken announcement for a detected object and position.

    Only announces object label and relative image position. Does not speak
    estimated distance, maintaining reliable and concise audio guidance.

    Examples:
        - 'Person on your left.'
        - 'Chair in front of you.'
        - 'Bottle on your right.'
    """
    clean_name = class_name.strip()
    formatted_name = clean_name[:1].upper() + clean_name[1:] if clean_name else "Object"
    pos_phrase = position.phrase
    return f"{formatted_name} {pos_phrase}."


def format_multi_object_announcement(
    items: List[Tuple[str, HorizontalPosition, Optional[float]]],
    max_items: int = 3,
) -> str:
    """Builds a combined, concise announcement for multiple objects detected simultaneously.

    Speaks only object names and relative positions without distance numbers.

    Example:
        'Multiple objects detected: person on your left, chair in front of you, bottle on your right.'
    """
    if not items:
        return ""
    if len(items) == 1:
        name, pos, dist = items[0]
        return format_position_announcement(name, pos, dist)

    selected = items[:max_items]
    clauses = []
    for name, pos, _ in selected:
        c_name = name.lower().strip()
        pos_phrase = pos.phrase
        clauses.append(f"{c_name} {pos_phrase}")

    combined_clauses = ", ".join(clauses)
    return f"Multiple objects detected: {combined_clauses}."
