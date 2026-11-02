"""Nearest-object prioritization, temporal distance smoothing, and distance-based voice guidance.

SAFETY DISCLAIMER:
Monocular depth estimation is an assistive estimation technique and NOT a safety-certified
ranging or collision prevention system. Distance estimations are approximate indications
(reported as 'approximately X metres') and must not be treated as certified measurements.
Never provide navigation or movement instructions based on these estimates.
"""

from __future__ import annotations

import math
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from config.settings import AlertConfig, DepthConfig
from detection.detector import DetectionResult
from detection.position import HorizontalPosition, classify_horizontal_position
from utils.logging_config import get_logger

logger = get_logger("nearest_object")


class WarningDistanceLevel(str, Enum):
    """Distance-based warning classification levels."""
    URGENT = "URGENT"    # Under 1.0 metre: High alert proximity
    CAUTION = "CAUTION"  # 1.0 to 2.0 metres: Medium caution proximity
    NORMAL = "NORMAL"    # Over 2.0 metres: Normal situational awareness
    UNKNOWN = "UNKNOWN"  # Distance unavailable or unreliable

    @classmethod
    def classify(
        cls,
        distance: Optional[float],
        urgent_threshold: float = 1.0,
        caution_threshold: float = 2.0,
    ) -> WarningDistanceLevel:
        """Categorizes an estimated metric distance into a warning level."""
        if distance is None or not math.isfinite(distance) or distance <= 0.0:
            return cls.UNKNOWN
        if distance < urgent_threshold:
            return cls.URGENT
        elif distance < caution_threshold:
            return cls.CAUTION
        else:
            return cls.NORMAL


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class TrackedObject:
    """Maintains identity, temporal smoothing, and distance history for an object."""

    def __init__(
        self,
        track_id: int,
        detection: DetectionResult,
        alpha: float = 0.35,
    ) -> None:
        self.track_id = track_id
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        self.box = detection.box
        self.alpha = alpha

        self.raw_distance = detection.distance
        self.smoothed_distance: Optional[float] = detection.distance
        self.history_distances: deque[float] = deque(maxlen=6)
        if detection.distance is not None:
            self.history_distances.append(detection.distance)

        self.last_seen_time = detection.timestamp
        self.seen_frames = 1
        self.position = detection.position or "Center"
        self.relative_x = detection.relative_x or 0.5
        self.warning_level = WarningDistanceLevel.classify(self.smoothed_distance)

        # Announcement tracking
        self.last_alert_time: Optional[float] = None
        self.last_announced_distance: Optional[float] = None
        self.last_announced_warning_level: Optional[WarningDistanceLevel] = None
        self.last_announced_position: Optional[str] = None

    def update(self, detection: DetectionResult, current_time: float) -> None:
        """Updates tracked state with a new frame detection."""
        self.box = detection.box
        self.last_seen_time = current_time
        self.seen_frames += 1
        if detection.position:
            self.position = detection.position
        if detection.relative_x is not None:
            self.relative_x = detection.relative_x

        raw = detection.distance
        self.raw_distance = raw

        if raw is not None and math.isfinite(raw) and raw > 0.0:
            self.history_distances.append(raw)
            if self.smoothed_distance is None:
                self.smoothed_distance = raw
            else:
                # Exponential moving average filter
                self.smoothed_distance = round(
                    self.alpha * raw + (1.0 - self.alpha) * self.smoothed_distance,
                    2,
                )
        else:
            # If raw distance temporarily missing in this frame, retain smoothed for up to 2 frames
            if len(self.history_distances) == 0:
                self.smoothed_distance = None

        self.warning_level = WarningDistanceLevel.classify(self.smoothed_distance)


class NearestObjectManager:
    """Manages closest-object identification, distance smoothing, hysteresis, and voice alerts."""

    def __init__(
        self,
        alert_config: AlertConfig,
        depth_config: DepthConfig,
    ) -> None:
        self.alert_config = alert_config
        self.depth_config = depth_config

        self._next_track_id = 1
        self._tracks: Dict[int, TrackedObject] = {}
        self._current_nearest_track_id: Optional[int] = None

        # Tracking and hysteresis parameters
        raw_h = getattr(depth_config, "hysteresis_distance_m", 0.25)
        raw_s = getattr(depth_config, "significant_distance_delta_m", 0.5)
        raw_a = getattr(depth_config, "temporal_smoothing_alpha", 0.35)
        self.hysteresis_distance_m = float(str(raw_h).split("#")[0].strip())
        self.significant_distance_delta_m = float(str(raw_s).split("#")[0].strip())
        self.smoothing_alpha = float(str(raw_a).split("#")[0].strip())
        self.track_timeout_seconds = 1.2
        self.min_frames_to_switch = 2

        # Alert interval settings
        self.announcement_interval_seconds = getattr(alert_config, "announcement_interval_seconds", 3.0)
        self.cooldown_seconds = getattr(alert_config, "cooldown_seconds", 4.0)
        self.urgent_cooldown_seconds = 1.5

        # Global alert timing
        self._last_global_alert_time: float = 0.0
        self._last_announced_phrase: str = ""

    def reset(self) -> None:
        """Clears all tracking and announcement history."""
        self._tracks.clear()
        self._current_nearest_track_id = None
        self._last_global_alert_time = 0.0
        self._last_announced_phrase = ""

    def _match_and_update_tracks(
        self,
        detections: List[DetectionResult],
        current_time: float,
    ) -> List[Tuple[DetectionResult, TrackedObject]]:
        """Matches current detections to existing tracked objects using IoU and class."""
        matched_pairs: List[Tuple[DetectionResult, TrackedObject]] = []
        unmatched_dets = list(detections)

        # Match existing tracks with highest IoU detection of same class
        for track_id, track in list(self._tracks.items()):
            best_det_idx = -1
            best_iou = 0.20  # Minimum IoU threshold

            for idx, det in enumerate(unmatched_dets):
                if det.class_name.lower() == track.class_name.lower():
                    iou = compute_iou(track.box, det.box)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = idx

            if best_det_idx >= 0:
                det = unmatched_dets.pop(best_det_idx)
                track.update(det, current_time)
                matched_pairs.append((det, track))

        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            new_track = TrackedObject(
                track_id=self._next_track_id,
                detection=det,
                alpha=self.smoothing_alpha,
            )
            self._tracks[self._next_track_id] = new_track
            matched_pairs.append((det, new_track))
            self._next_track_id += 1

        # Prune dead tracks that have disappeared beyond timeout
        dead_ids = [
            tid for tid, trk in self._tracks.items()
            if (current_time - trk.last_seen_time) > self.track_timeout_seconds
        ]
        for tid in dead_ids:
            if self._current_nearest_track_id == tid:
                self._current_nearest_track_id = None
            self._tracks.pop(tid, None)

        return matched_pairs

    def process_frame(
        self,
        detections: List[DetectionResult],
        current_time: Optional[float] = None,
        frame_width: Optional[int] = None,
    ) -> Tuple[List[DetectionResult], Optional[DetectionResult]]:
        """Updates object tracking, selects nearest object with hysteresis, and enriches detections.

        Returns:
            Tuple of:
            - enriched_detections: List of DetectionResult objects with smoothed distance,
              warning level, and is_closest flag set.
            - nearest_detection: The primary closest DetectionResult, or None if no detections.
        """
        now = current_time if current_time is not None else time.time()
        w = frame_width or self.alert_config.frame_width

        if not detections:
            # Prune old tracks
            dead_ids = [
                tid for tid, trk in self._tracks.items()
                if (now - trk.last_seen_time) > self.track_timeout_seconds
            ]
            for tid in dead_ids:
                if self._current_nearest_track_id == tid:
                    self._current_nearest_track_id = None
                self._tracks.pop(tid, None)
            return [], None

        # 1. Update tracks with current detections
        matched_pairs = self._match_and_update_tracks(detections, now)

        # 2. Identify candidates with valid smoothed distances
        valid_candidates: List[TrackedObject] = [
            trk for _, trk in matched_pairs
            if trk.smoothed_distance is not None and trk.smoothed_distance > 0.0
        ]

        # 3. Select nearest object using hysteresis
        chosen_nearest_track: Optional[TrackedObject] = None

        if valid_candidates:
            # Sort candidate tracks by distance ascending (closest first)
            valid_candidates.sort(key=lambda t: t.smoothed_distance)  # type: ignore

            current_nearest = self._tracks.get(self._current_nearest_track_id) if self._current_nearest_track_id else None

            if current_nearest is not None and current_nearest.smoothed_distance is not None and current_nearest in valid_candidates:
                # Active nearest object is still detected and valid.
                best_new_candidate = valid_candidates[0]
                if best_new_candidate.track_id != current_nearest.track_id:
                    # Only switch if new candidate is closer by at least hysteresis_distance_m
                    # and has been confirmed across frames
                    curr_d = current_nearest.smoothed_distance
                    new_d = best_new_candidate.smoothed_distance
                    if (curr_d - new_d) >= self.hysteresis_distance_m and best_new_candidate.seen_frames >= self.min_frames_to_switch:
                        chosen_nearest_track = best_new_candidate
                        self._current_nearest_track_id = best_new_candidate.track_id
                        logger.debug("Switched nearest object to track %d (dist: %.2fm < %.2fm - %.2fm)",
                                     best_new_candidate.track_id, new_d, curr_d, self.hysteresis_distance_m)
                    else:
                        chosen_nearest_track = current_nearest
                else:
                    chosen_nearest_track = current_nearest
            else:
                # Previous nearest is gone or invalid; select top candidate immediately
                chosen_nearest_track = valid_candidates[0]
                self._current_nearest_track_id = chosen_nearest_track.track_id
        else:
            # No valid distances; fallback to highest-confidence detection if available
            self._current_nearest_track_id = None
            if matched_pairs:
                matched_pairs_sorted = sorted(matched_pairs, key=lambda p: p[0].confidence, reverse=True)
                chosen_nearest_track = matched_pairs_sorted[0][1]

        # 4. Construct enriched DetectionResult instances
        enriched_results: List[DetectionResult] = []
        nearest_det: Optional[DetectionResult] = None

        for det, track in matched_pairs:
            is_closest = (chosen_nearest_track is not None and track.track_id == chosen_nearest_track.track_id)
            dist_val = track.smoothed_distance if track.smoothed_distance is not None else det.distance
            avail = (dist_val is not None and dist_val > 0.0)
            lvl = track.warning_level.value if avail else None

            enriched = DetectionResult(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                box=det.box,
                timestamp=det.timestamp,
                distance=dist_val,
                position=track.position,
                relative_x=track.relative_x,
                is_closest=is_closest,
                warning_level=lvl,
                distance_available=avail,
            )
            enriched_results.append(enriched)
            if is_closest:
                nearest_det = enriched

        return enriched_results, nearest_det

    def format_class_name(self, raw_name: str) -> str:
        """Formats class name using friendly spoken mapping."""
        c_lower = raw_name.lower().strip()
        if hasattr(self.alert_config, "spoken_class_mapping") and c_lower in self.alert_config.spoken_class_mapping:
            return self.alert_config.spoken_class_mapping[c_lower]
        return c_lower.replace("_", " ").title()

    def format_direction_phrase(self, position_label: Optional[str]) -> str:
        """Translates position label to natural spoken direction phrase."""
        pos = (position_label or "Center").upper()
        if "LEFT" in pos:
            return "on your left"
        elif "RIGHT" in pos:
            return "on your right"
        return "in front of you"

    def format_nearest_voice_alert(self, nearest_det: DetectionResult) -> str:
        """Generates natural, accessible spoken English announcement with distance and direction.

        Examples:
            - Under 1m: 'Warning! Backpack on your right, approximately 0.7 metres away. It is the closest detected object.'
            - 1-2m:     'Chair in front of you, approximately 1.2 metres away.'
            - > 2m:     'Person on your left, approximately 2.5 metres away.'
            - Distance unavailable: 'Person on your left, distance unavailable.'
        """
        cname = self.format_class_name(nearest_det.class_name)
        direction = self.format_direction_phrase(nearest_det.position)

        if nearest_det.distance is not None and nearest_det.distance_available:
            dist_str = f"{nearest_det.distance:.1f}"
            lvl = nearest_det.warning_level or WarningDistanceLevel.classify(nearest_det.distance).value

            if lvl == WarningDistanceLevel.URGENT.value:
                return f"Warning! {cname} {direction}, approximately {dist_str} metres away. It is the closest detected object."
            else:
                return f"{cname} {direction}, approximately {dist_str} metres away."
        else:
            return f"{cname} {direction}, distance unavailable."

    def evaluate_voice_alert(
        self,
        nearest_det: Optional[DetectionResult],
        current_time: Optional[float] = None,
    ) -> Optional[Tuple[str, DetectionResult, bool]]:
        """Evaluates whether an announcement should be spoken for the closest object.

        Triggers announcement on:
        1. New closest object appearance.
        2. Escalation to URGENT warning level (< 1.0m).
        3. Significant change in distance (>= significant_distance_delta_m, default 0.5m).
        4. Significant change in direction/position.
        5. Expiration of continuous announcement interval (default 3.0s).

        Returns:
            Tuple of (announcement_text, nearest_detection, is_urgent) or None.
        """
        if nearest_det is None:
            return None

        now = current_time if current_time is not None else time.time()
        track = self._tracks.get(self._current_nearest_track_id) if self._current_nearest_track_id else None

        phrase = self.format_nearest_voice_alert(nearest_det)
        is_urgent = (nearest_det.warning_level == WarningDistanceLevel.URGENT.value)

        # Check cooldowns
        cooldown = self.urgent_cooldown_seconds if is_urgent else self.cooldown_seconds
        time_since_last_alert = now - self._last_global_alert_time

        is_due = False
        reason = ""

        if track is None or track.last_alert_time is None:
            # 1. New nearest object
            if time_since_last_alert >= (1.0 if not is_urgent else 0.5):
                is_due = True
                reason = "new_object"
        else:
            # 2. Escalation to URGENT
            if is_urgent and track.last_announced_warning_level != WarningDistanceLevel.URGENT:
                if time_since_last_alert >= self.urgent_cooldown_seconds:
                    is_due = True
                    reason = "urgent_escalation"

            # 3. Significant distance change
            elif (
                track.last_announced_distance is not None
                and nearest_det.distance is not None
                and abs(nearest_det.distance - track.last_announced_distance) >= self.significant_distance_delta_m
            ):
                if time_since_last_alert >= cooldown:
                    is_due = True
                    reason = "distance_delta"

            # 4. Position / direction changed
            elif (
                track.last_announced_position is not None
                and nearest_det.position != track.last_announced_position
            ):
                if time_since_last_alert >= cooldown:
                    is_due = True
                    reason = "position_change"

            # 5. Continuous interval repetition
            elif (now - track.last_alert_time) >= self.announcement_interval_seconds:
                if time_since_last_alert >= cooldown:
                    is_due = True
                    reason = "interval_repeat"

        if not is_due:
            return None

        # Record announcement history
        self._last_global_alert_time = now
        self._last_announced_phrase = phrase

        if track is not None:
            track.last_alert_time = now
            track.last_announced_distance = nearest_det.distance
            track.last_announced_warning_level = WarningDistanceLevel.classify(nearest_det.distance)
            track.last_announced_position = nearest_det.position

        logger.info("Voice guidance triggered [%s%s]: '%s'",
                    reason, ", URGENT" if is_urgent else "", phrase)

        return (phrase, nearest_det, is_urgent)
