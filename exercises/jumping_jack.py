"""
exercises/jumping_jack.py

Purpose:
    Detects jumping jacks and counts repetitions by tracking two
    things together: whether the arms are raised (wrists above
    shoulders) and whether the legs are spread apart (ankle-to-ankle
    distance vs hip width). A rep only counts when both happen
    together and then both return together — raising your arms alone,
    or just stepping your feet apart alone, won't register as one.

Why shoulder/wrist and hip/ankle distances, not a joint angle:
    Squats, push-ups, and curls are single-joint bend movements, so a
    three-point angle at that joint is the natural measurement. A
    jumping jack isn't a joint bend — it's a whole-limb position
    change (arms up vs down, legs apart vs together), so relative
    position/distance fits better than an angle.
"""

import math
from enum import Enum
from typing import Optional

from pose.landmarks import (
    Landmark,
    get_hip_center,
    get_landmark,
    get_shoulder_center,
    is_visible,
)
from utils.smoothing import ExponentialMovingAverage


class JackState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"


def _distance(point_a, point_b) -> float:
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


class JumpingJackCounter:
    """
    Stateful jumping jack rep counter. Create one instance per workout
    session and call update() once per frame with the current landmark
    coordinates.

    Uses two independent hysteresis measurements combined with AND
    logic, rather than a single threshold:
      - arm_raise_ratio: how far the wrists sit above the shoulders,
        normalized by torso length. Rising above `arm_raised_ratio`
        means "arms up"; falling below `arm_lowered_ratio` means "arms
        down". The gap between the two is the hysteresis dead zone
        (see exercises/squat.py for why that prevents double counting).
      - leg_spread_ratio: ankle-to-ankle distance divided by hip
        width. Above `legs_open_ratio` means "legs apart"; below
        `legs_closed_ratio` means "legs together".

    The body only flips to OPEN when arms are up AND legs are apart,
    and only flips back to CLOSED when arms are down AND legs are
    together — so a half-movement (e.g. arms up but feet never moved)
    can't trigger a state change.
    """

    def __init__(
        self,
        arm_raised_ratio: float = 0.3,
        arm_lowered_ratio: float = 0.1,
        legs_open_ratio: float = 1.7,
        legs_closed_ratio: float = 1.3,
        min_visibility: float = 0.5,
        smoothing_alpha: float = 0.5,
    ):
        if arm_lowered_ratio >= arm_raised_ratio:
            raise ValueError("arm_lowered_ratio must be lower than arm_raised_ratio")
        if legs_closed_ratio >= legs_open_ratio:
            raise ValueError("legs_closed_ratio must be lower than legs_open_ratio")

        self.arm_raised_ratio = arm_raised_ratio
        self.arm_lowered_ratio = arm_lowered_ratio
        self.legs_open_ratio = legs_open_ratio
        self.legs_closed_ratio = legs_closed_ratio
        self.min_visibility = min_visibility

        self.state = JackState.CLOSED
        self.rep_count = 0
        self.current_arm_ratio: Optional[float] = None
        self.current_leg_ratio: Optional[float] = None
        # Two independent smoothers - arms and legs are different
        # signals with their own noise, so blending them together
        # would let noise on one limb bleed into the other's reading.
        self._arm_smoother = ExponentialMovingAverage(alpha=smoothing_alpha)
        self._leg_smoother = ExponentialMovingAverage(alpha=smoothing_alpha)

    def update(self, coordinates: dict) -> dict:
        """
        Process one frame's landmark coordinates. Returns a status
        dict:
            {
                "rep_count": int,
                "state": "CLOSED" | "OPEN",
                "arm_raise_ratio": float | None,
                "leg_spread_ratio": float | None,
                "landmarks_visible": bool,
            }
        """
        arm_ratio = self._compute_arm_raise_ratio(coordinates)
        leg_ratio = self._compute_leg_spread_ratio(coordinates)

        if arm_ratio is None or leg_ratio is None:
            # Don't touch the smoothers or current_*_ratio here - leave
            # them holding their last known values, same as every other
            # counter does when a frame's landmarks aren't visible.
            return self._status(landmarks_visible=False)

        smoothed_arm_ratio = self._arm_smoother.update(arm_ratio)
        smoothed_leg_ratio = self._leg_smoother.update(leg_ratio)

        self.current_arm_ratio = smoothed_arm_ratio
        self.current_leg_ratio = smoothed_leg_ratio

        self._update_state(smoothed_arm_ratio, smoothed_leg_ratio)
        return self._status(landmarks_visible=True)

    def _compute_arm_raise_ratio(self, coordinates: dict) -> Optional[float]:
        """
        How far the wrists are above the shoulders, normalized by
        torso length (shoulder-center to hip-center distance) so it
        works regardless of distance from the camera. Positive means
        wrists above shoulders; near zero or negative means arms at or
        below shoulder height. Averages both arms when both are
        visible; falls back to whichever single arm is visible.
        """
        shoulder_center = get_shoulder_center(coordinates)
        hip_center = get_hip_center(coordinates)
        if not is_visible(shoulder_center, self.min_visibility) or not is_visible(hip_center, self.min_visibility):
            return None

        torso_length = _distance(shoulder_center, hip_center)
        if torso_length == 0:
            return None

        left_shoulder = get_landmark(coordinates, Landmark.LEFT_SHOULDER)
        left_wrist = get_landmark(coordinates, Landmark.LEFT_WRIST)
        right_shoulder = get_landmark(coordinates, Landmark.RIGHT_SHOULDER)
        right_wrist = get_landmark(coordinates, Landmark.RIGHT_WRIST)

        ratios = []
        if is_visible(left_shoulder, self.min_visibility) and is_visible(left_wrist, self.min_visibility):
            ratios.append((left_shoulder.y - left_wrist.y) / torso_length)
        if is_visible(right_shoulder, self.min_visibility) and is_visible(right_wrist, self.min_visibility):
            ratios.append((right_shoulder.y - right_wrist.y) / torso_length)

        if not ratios:
            return None
        return sum(ratios) / len(ratios)

    def _compute_leg_spread_ratio(self, coordinates: dict) -> Optional[float]:
        """
        Ankle-to-ankle distance divided by hip-to-hip distance. Close
        to 1.0-1.3 when standing normally (ankles roughly under the
        hips), rising well above that when the legs spread apart.
        """
        left_hip = get_landmark(coordinates, Landmark.LEFT_HIP)
        right_hip = get_landmark(coordinates, Landmark.RIGHT_HIP)
        left_ankle = get_landmark(coordinates, Landmark.LEFT_ANKLE)
        right_ankle = get_landmark(coordinates, Landmark.RIGHT_ANKLE)

        required_points = (left_hip, right_hip, left_ankle, right_ankle)
        if any(not is_visible(point, self.min_visibility) for point in required_points):
            return None

        hip_width = _distance(left_hip, right_hip)
        if hip_width == 0:
            return None

        ankle_spread = _distance(left_ankle, right_ankle)
        return ankle_spread / hip_width

    def _update_state(self, arm_ratio: float, leg_ratio: float) -> None:
        arms_up = arm_ratio > self.arm_raised_ratio
        arms_down = arm_ratio < self.arm_lowered_ratio
        legs_open = leg_ratio > self.legs_open_ratio
        legs_closed = leg_ratio < self.legs_closed_ratio

        if self.state == JackState.CLOSED and arms_up and legs_open:
            self.state = JackState.OPEN

        elif self.state == JackState.OPEN and arms_down and legs_closed:
            self.state = JackState.CLOSED
            self.rep_count += 1

    def _status(self, landmarks_visible: bool) -> dict:
        return {
            "rep_count": self.rep_count,
            "state": self.state.value,
            "arm_raise_ratio": self.current_arm_ratio,
            "leg_spread_ratio": self.current_leg_ratio,
            "landmarks_visible": landmarks_visible,
        }

    def reset(self) -> None:
        """Reset the counter back to a fresh session (state CLOSED, count 0)."""
        self.state = JackState.CLOSED
        self.rep_count = 0
        self.current_arm_ratio = None
        self.current_leg_ratio = None
        self._arm_smoother.reset()
        self._leg_smoother.reset()