"""
exercises/bicep_curl.py

Purpose:
    Detects bicep curls and counts repetitions using the elbow angle
    (shoulder-elbow-wrist), tracking EXTENDED/CONTRACTED states with
    the same hysteresis pattern used for squats and push-ups (see
    exercises/squat.py for the full explanation of why two separated
    thresholds prevent double counting).

    Also gives two basic form checks specific to curls:
      1. Elbow drift — the most common curl mistake is swinging the
         whole upper arm forward to use momentum ("cheating" the curl)
         instead of keeping the elbow fixed and only rotating the
         forearm. We track how far the elbow strays horizontally from
         directly under the shoulder, scaled by torso length so it
         works regardless of how close the camera is.
      2. Contraction depth — flags a curl that never bent far enough
         to count as a full-range rep.

Why elbow angle:
    Arm hanging straight at the bottom: angle near 180 deg (extended).
    Arm fully curled at the top: angle drops to roughly 30-50 deg
    (contracted) for most people's range of motion.
"""

import math
from enum import Enum
from typing import Optional

from pose.landmarks import ARM_LANDMARKS, Landmark, get_arm_points, get_landmark, is_visible, pick_more_visible_side
from utils.angles import calculate_angle_or_none
from utils.smoothing import ExponentialMovingAverage


class CurlState(Enum):
    EXTENDED = "EXTENDED"
    CONTRACTED = "CONTRACTED"


def _distance(point_a, point_b) -> float:
    """Straight-line pixel distance between two landmark points."""
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


class BicepCurlCounter:
    """
    Stateful bicep curl rep counter with basic form feedback. Create
    one instance per workout session and call update() once per frame
    with the current landmark coordinates.
    """

    def __init__(
        self,
        contracted_threshold: float = 50.0,
        extended_threshold: float = 160.0,
        good_contraction_threshold: float = 60.0,
        max_elbow_drift_ratio: float = 0.3,
        min_visibility: float = 0.5,
        smoothing_alpha: float = 0.5,
    ):
        if contracted_threshold >= extended_threshold:
            raise ValueError("contracted_threshold must be lower than extended_threshold")

        self.contracted_threshold = contracted_threshold
        self.extended_threshold = extended_threshold
        self.good_contraction_threshold = good_contraction_threshold
        self.max_elbow_drift_ratio = max_elbow_drift_ratio
        self.min_visibility = min_visibility

        self.state = CurlState.EXTENDED
        self.rep_count = 0
        self.current_elbow_angle: Optional[float] = None
        self.active_side: Optional[str] = None
        self._min_angle_this_rep: Optional[float] = None
        self._max_drift_ratio_this_rep: float = 0.0
        self.last_rep_feedback: Optional[str] = None
        self._angle_smoother = ExponentialMovingAverage(alpha=smoothing_alpha)

    def update(self, coordinates: dict) -> dict:
        """
        Process one frame's landmark coordinates. Returns a status
        dict:
            {
                "rep_count": int,
                "state": "EXTENDED" | "CONTRACTED",
                "elbow_angle": float | None,
                "side": "left" | "right" | None,
                "landmarks_visible": bool,
                "last_rep_feedback": str | None,  # set once per completed rep
            }
        """
        side = pick_more_visible_side(coordinates, ARM_LANDMARKS)
        shoulder, elbow, wrist = get_arm_points(coordinates, side)

        landmarks_visible = (
            is_visible(shoulder, self.min_visibility)
            and is_visible(elbow, self.min_visibility)
            and is_visible(wrist, self.min_visibility)
        )
        if not landmarks_visible:
            return self._status(landmarks_visible=False)

        elbow_angle = calculate_angle_or_none(shoulder, elbow, wrist)
        if elbow_angle is None:
            return self._status(landmarks_visible=False)

        smoothed_angle = self._angle_smoother.update(elbow_angle)

        self.current_elbow_angle = smoothed_angle
        self.active_side = side

        self._track_elbow_drift(coordinates, side, shoulder, elbow)
        self._update_state(smoothed_angle)

        return self._status(landmarks_visible=True)

    def _track_elbow_drift(self, coordinates: dict, side: str, shoulder, elbow) -> None:
        """
        Update the largest elbow-drift ratio seen so far this rep.
        Drift ratio = horizontal elbow-to-shoulder distance / torso
        length, so it's scale-independent (works whether you're close
        to or far from the camera).
        """
        hip_landmark = Landmark.LEFT_HIP if side == "left" else Landmark.RIGHT_HIP
        hip = get_landmark(coordinates, hip_landmark)

        if not is_visible(hip, self.min_visibility):
            return

        torso_length = _distance(shoulder, hip)
        if torso_length == 0:
            return

        horizontal_drift = abs(elbow.x - shoulder.x)
        drift_ratio = horizontal_drift / torso_length

        if drift_ratio > self._max_drift_ratio_this_rep:
            self._max_drift_ratio_this_rep = drift_ratio

    def _update_state(self, elbow_angle: float) -> None:
        """Same hysteresis pattern as SquatCounter, applied to curl direction."""
        if self.state == CurlState.EXTENDED and elbow_angle < self.contracted_threshold:
            self.state = CurlState.CONTRACTED
            self._min_angle_this_rep = elbow_angle

        elif self.state == CurlState.CONTRACTED:
            if self._min_angle_this_rep is None or elbow_angle < self._min_angle_this_rep:
                self._min_angle_this_rep = elbow_angle

            if elbow_angle > self.extended_threshold:
                self.state = CurlState.EXTENDED
                self.rep_count += 1
                self._finalize_rep_feedback()
                self._min_angle_this_rep = None
                self._max_drift_ratio_this_rep = 0.0

    def _finalize_rep_feedback(self) -> None:
        """Pick one feedback message for the rep that just completed."""
        if self._max_drift_ratio_this_rep > self.max_elbow_drift_ratio:
            self.last_rep_feedback = "Keep your elbow steady - avoid swinging your upper arm"
        elif self._min_angle_this_rep is not None and self._min_angle_this_rep > self.good_contraction_threshold:
            self.last_rep_feedback = "Curl higher for a full contraction"
        else:
            self.last_rep_feedback = "Good rep"

    def _status(self, landmarks_visible: bool) -> dict:
        return {
            "rep_count": self.rep_count,
            "state": self.state.value,
            "elbow_angle": self.current_elbow_angle,
            "side": self.active_side,
            "landmarks_visible": landmarks_visible,
            "last_rep_feedback": self.last_rep_feedback,
        }

    def reset(self) -> None:
        """Reset the counter back to a fresh session (state EXTENDED, count 0)."""
        self.state = CurlState.EXTENDED
        self.rep_count = 0
        self.current_elbow_angle = None
        self.active_side = None
        self._min_angle_this_rep = None
        self._max_drift_ratio_this_rep = 0.0
        self.last_rep_feedback = None
        self._angle_smoother.reset()