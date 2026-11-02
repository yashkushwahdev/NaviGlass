"""Object filtering, deduplication, relative position classification, and alert decision logic."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

from config.settings import AlertConfig
from detection.detector import DetectionResult
from detection.position import (
    HorizontalPosition,
    classify_horizontal_position,
    format_multi_object_announcement,
    format_position_announcement,
)
from utils.logging_config import get_logger

logger = get_logger("filtering")


class DetectionFilter:
    """Filters detections, tracks visible objects, manages position changes, and schedules periodic announcements."""

    def __init__(self, config: AlertConfig, min_confidence: float = 0.35) -> None:
        self.config = config
        self.min_confidence = min_confidence
        self._eligible_classes: Set[str] = set(c.lower() for c in config.eligible_classes)

        # Configurable announcement interval for continuously visible objects
        self.announcement_interval_seconds: float = getattr(config, "announcement_interval_seconds", 3.0)
        self.cooldown_seconds: float = getattr(config, "cooldown_seconds", 4.0)

        # Timeout to consider an object disappeared from camera view (seconds)
        self.disappearance_timeout_seconds: float = 1.5

        # Tracking state
        self._last_alert_time: Dict[str, float] = {}
        self._last_seen_time: Dict[str, float] = {}
        self._last_positions: Dict[str, HorizontalPosition] = {}
        self._last_announced_position: Dict[str, HorizontalPosition] = {}
        self._last_relative_x: Dict[str, float] = {}

    def filter_detections(
        self,
        detections: List[DetectionResult],
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Filters detections by confidence threshold and eligibility."""
        thresh = confidence_threshold if confidence_threshold is not None else self.min_confidence

        filtered: List[DetectionResult] = []
        for d in detections:
            if d.confidence < thresh:
                continue
            cname = d.class_name.lower()
            if self._eligible_classes and cname not in self._eligible_classes:
                continue
            filtered.append(d)

        return filtered

    def deduplicate(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Deduplicates multiple detections of the same class in a single frame.

        Keeps the highest-confidence detection per class.
        """
        best_by_class: Dict[str, DetectionResult] = {}
        for d in detections:
            cname = d.class_name.lower()
            if cname not in best_by_class or d.confidence > best_by_class[cname].confidence:
                best_by_class[cname] = d

        return list(best_by_class.values())

    def format_class_name(self, raw_class_name: str) -> str:
        """Translates raw class name to human-friendly spoken representation."""
        c_lower = raw_class_name.lower()
        if c_lower in self.config.spoken_class_mapping:
            return self.config.spoken_class_mapping[c_lower]
        return c_lower.replace("_", " ").title()

    def classify_position(
        self,
        detection: DetectionResult,
        frame_width: Optional[int] = None,
    ) -> Tuple[HorizontalPosition, float]:
        """Classifies the horizontal relative position of a detected object."""
        w = frame_width or self.config.frame_width
        pos_cfg = self.config.position
        cname = detection.class_name.lower()
        prev_pos = self._last_positions.get(cname)

        return classify_horizontal_position(
            box=detection.box,
            frame_width=w,
            left_threshold=pos_cfg.left_threshold,
            right_threshold=pos_cfg.right_threshold,
            current_position=prev_pos,
            hysteresis=pos_cfg.hysteresis,
        )

    def attach_position_metadata(
        self,
        detection: DetectionResult,
        frame_width: Optional[int] = None,
    ) -> DetectionResult:
        """Returns an enriched copy of DetectionResult with position and relative_x metadata."""
        pos, rel_x = self.classify_position(detection, frame_width=frame_width)
        return DetectionResult(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            box=detection.box,
            timestamp=detection.timestamp,
            distance=detection.distance,
            position=pos.label,
            relative_x=rel_x,
        )

    def format_announcement(
        self,
        candidate: DetectionResult,
        position: HorizontalPosition,
        distance: Optional[float] = None,
    ) -> str:
        """Formats spoken announcement text (strictly label + relative position, no distance)."""
        spoken_name = self.format_class_name(candidate.class_name)
        fmt = self.config.announcement_format

        # If user explicitly disabled position in config
        if not self.config.include_position and not self.config.position.enabled:
            return fmt.format(class_name=spoken_name, position="")

        if "{position}" in fmt:
            pos_phrase = position.phrase
            return fmt.format(class_name=spoken_name, position=pos_phrase).strip()

        return format_position_announcement(spoken_name, position, distance=None)

    def _update_visibility_and_prune(self, current_classes: Set[str], current_time: float) -> None:
        """Updates last seen timestamps for current objects and cleans up state for absent objects."""
        # Update seen timestamps for currently detected objects
        for cname in current_classes:
            self._last_seen_time[cname] = current_time

        # Check for absent objects that have exceeded the disappearance timeout
        disappeared = [
            cname for cname, seen_t in self._last_seen_time.items()
            if cname not in current_classes and (current_time - seen_t) > self.disappearance_timeout_seconds
        ]
        for cname in disappeared:
            self._last_seen_time.pop(cname, None)
            self._last_alert_time.pop(cname, None)
            self._last_positions.pop(cname, None)
            self._last_announced_position.pop(cname, None)
            self._last_relative_x.pop(cname, None)
            logger.debug("Pruned disappeared object '%s'", cname)

    def evaluate_alert(
        self,
        detections: List[DetectionResult],
        current_time: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        frame_width: Optional[int] = None,
    ) -> Optional[Tuple[str, DetectionResult]]:
        """Evaluates detected objects and returns an announcement when appropriate.

        Triggers an announcement:
        1. When a new object appears in view.
        2. When an existing object significantly changes position (e.g. left -> center)
           after a short cooldown.
        3. At the configured announcement_interval_seconds for continuously visible objects.

        Returns:
            Tuple of (announcement_text, primary_detection_result) or None.
        """
        now = current_time if current_time is not None else time.time()
        w = frame_width or self.config.frame_width

        # 1. Filter and deduplicate current frame detections
        candidates = self.filter_detections(detections, confidence_threshold=confidence_threshold)
        current_classes = set(c.class_name.lower() for c in candidates)

        # Update visibility and prune objects that disappeared
        self._update_visibility_and_prune(current_classes, now)

        if not candidates:
            return None

        unique_candidates = self.deduplicate(candidates)

        # Sort candidates: highest confidence first
        unique_candidates.sort(key=lambda d: d.confidence, reverse=True)

        interval = self.announcement_interval_seconds
        pos_change_cooldown = self.config.position.position_change_cooldown_seconds

        due_candidates: List[Tuple[DetectionResult, HorizontalPosition, float, bool]] = []

        for candidate in unique_candidates:
            cname = candidate.class_name.lower()
            pos, rel_x = self.classify_position(candidate, frame_width=w)

            last_spoken = self._last_alert_time.get(cname)
            last_announced = self._last_announced_position.get(cname)

            position_changed = (last_announced is not None and pos != last_announced)

            is_due = False
            is_pos_shift = False

            if last_spoken is None:
                # Brand new appearance: announce immediately
                is_due = True
            elif position_changed and self.config.position.announce_position_change:
                # Position shifted: announce if position change cooldown has elapsed
                if (now - last_spoken) >= pos_change_cooldown:
                    is_due = True
                    is_pos_shift = True
            elif (now - last_spoken) >= interval:
                # Continuous visibility interval elapsed: repeat announcement
                is_due = True

            if is_due:
                due_candidates.append((candidate, pos, rel_x, is_pos_shift))

            # Update tracked position continuously
            self._last_positions[cname] = pos
            self._last_relative_x[cname] = rel_x

        if not due_candidates:
            return None

        # Check if multi-object announcement mode is active
        use_multi = getattr(self.config.position, "multi_object_announcements", False)
        if use_multi and len(due_candidates) > 1:
            items = [
                (self.format_class_name(c[0].class_name), c[1], None)
                for c in due_candidates
            ]
            announcement = format_multi_object_announcement(
                items,
                max_items=getattr(self.config.position, "max_multi_objects", 3),
            )
            for c, pos, rel_x, _ in due_candidates:
                c_lower = c.class_name.lower()
                self._last_alert_time[c_lower] = now
                self._last_announced_position[c_lower] = pos
            primary_c = due_candidates[0]
            enriched = self.attach_position_metadata(primary_c[0], frame_width=w)
            logger.info("Multi-object alert triggered: '%s'", announcement)
            return (announcement, enriched)

        # Single object announcement (highest priority candidate due)
        best_candidate, best_pos, best_rel_x, _ = due_candidates[0]
        cname = best_candidate.class_name.lower()
        self._last_alert_time[cname] = now
        self._last_announced_position[cname] = best_pos

        announcement = self.format_announcement(best_candidate, best_pos)
        enriched = DetectionResult(
            class_id=best_candidate.class_id,
            class_name=best_candidate.class_name,
            confidence=best_candidate.confidence,
            box=best_candidate.box,
            timestamp=best_candidate.timestamp,
            distance=best_candidate.distance,
            position=best_pos.label,
            relative_x=best_rel_x,
        )

        logger.info("Alert triggered: '%s' (conf: %.2f, rel_x: %.2f)", announcement, best_candidate.confidence, best_rel_x)
        return (announcement, enriched)

    def evaluate_multi_alert(
        self,
        detections: List[DetectionResult],
        current_time: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        frame_width: Optional[int] = None,
    ) -> Optional[Tuple[str, List[DetectionResult]]]:
        """Evaluates multiple detected objects simultaneously for a concise combined announcement."""
        now = current_time if current_time is not None else time.time()
        w = frame_width or self.config.frame_width

        candidates = self.filter_detections(detections, confidence_threshold=confidence_threshold)
        current_classes = set(c.class_name.lower() for c in candidates)
        self._update_visibility_and_prune(current_classes, now)

        if not candidates:
            return None

        unique_candidates = self.deduplicate(candidates)
        interval = self.announcement_interval_seconds
        pos_change_cooldown = self.config.position.position_change_cooldown_seconds

        due_items: List[Tuple[str, HorizontalPosition, Optional[float]]] = []
        due_dets: List[DetectionResult] = []

        for candidate in unique_candidates:
            cname = candidate.class_name.lower()
            pos, rel_x = self.classify_position(candidate, frame_width=w)
            enriched = self.attach_position_metadata(candidate, frame_width=w)

            last_spoken = self._last_alert_time.get(cname)
            last_announced = self._last_announced_position.get(cname)
            pos_changed = (last_announced is not None and pos != last_announced)

            if (
                last_spoken is None
                or (pos_changed and (now - last_spoken) >= pos_change_cooldown)
                or (now - last_spoken) >= interval
            ):
                due_items.append((self.format_class_name(cname), pos, None))
                due_dets.append(enriched)
                self._last_positions[cname] = pos
                self._last_announced_position[cname] = pos
                self._last_relative_x[cname] = rel_x
                self._last_alert_time[cname] = now

        if not due_items:
            return None

        announcement = format_multi_object_announcement(
            due_items,
            max_items=getattr(self.config.position, "max_multi_objects", 3),
        )
        return (announcement, due_dets)

    def reset(self) -> None:
        """Resets all cooldown and tracking timestamps."""
        self._last_alert_time.clear()
        self._last_seen_time.clear()
        self._last_positions.clear()
        self._last_announced_position.clear()
        self._last_relative_x.clear()
        logger.info("Alert cooldowns and history reset.")


# Backward compatibility alias
AlertManager = DetectionFilter
