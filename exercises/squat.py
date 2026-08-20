"""
exercises/squat.py

Purpose:
    Detects squat repetitions from the knee angle (hip-knee-ankle).
    Tracks a simple UP/DOWN state machine and counts one rep per
    complete DOWN-then-UP cycle.

Why a state machine, and why two thresholds instead of one:
    If we counted a rep every time the knee angle merely dipped below
    some single cutoff, tiny noise in pose detection near that cutoff
    would flicker back and forth across it and count several reps for
    one real squat. Two thresholds with a gap between them (a "dead
    zone") fix that:

        - Angle drops below DOWN_THRESHOLD (100 deg)  -> enter DOWN state
        - Angle rises above UP_THRESHOLD (160 deg)     -> enter UP state,
          and if we were previously DOWN, that's one completed rep

    Because the two thresholds are far apart, the knee angle has to
    travel through the whole dead zone before it can flip the state
    again, so a rep can only be counted once per genuine down-and-up
    movement — this is what "avoid double counting" means in practice.

    Hysteresis alone doesn't catch everything, though: a single bad
    frame (a momentary MediaPipe misdetection) can still swing the raw
    angle far enough to flip a state and register a phantom rep, even
    if the person never moved. The raw angle is smoothed with an
    exponential moving average (utils/smoothing.py) before it ever
    reaches the state machine, which damps exactly that kind of
    single-frame spike.
"""

from enum import Enum
from typing import Optional

from pose.landmarks import LEG_LANDMARKS, get_leg_points, is_visible, pick_more_visible_side
from utils.angles import calculate_angle_or_none
from utils.smoothing import ExponentialMovingAverage


class SquatState(Enum):
    UP = "UP"
    DOWN = "DOWN"


class SquatCounter:
    """
    Stateful squat rep counter. Create one instance per workout session
    and call update() once per frame with the current landmark
    coordinates.
    """

    def __init__(
        self,
        down_threshold: float = 100.0,
        up_threshold: float = 160.0,
        min_visibility: float = 0.5,
        smoothing_alpha: float = 0.5,
    ):
        if down_threshold >= up_threshold:
            raise ValueError("down_threshold must be lower than up_threshold")

        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.min_visibility = min_visibility

        self.state = SquatState.UP
        self.rep_count = 0
        self.current_knee_angle: Optional[float] = None
        self.active_side: Optional[str] = None
        self._angle_smoother = ExponentialMovingAverage(alpha=smoothing_alpha)

    def update(self, coordinates: dict) -> dict:
        """
        Process one frame's landmark coordinates. Updates internal
        state and rep_count as needed, and returns a status dict for
        display:
            {
                "rep_count": int,
                "state": "UP" | "DOWN",
                "knee_angle": float | None,
                "side": "left" | "right" | None,
                "landmarks_visible": bool,
            }

        If the hip/knee/ankle aren't clearly visible this frame, the
        state and count are left untouched rather than guessed at from
        unreliable data.
        """
        side = pick_more_visible_side(coordinates, LEG_LANDMARKS)
        hip, knee, ankle = get_leg_points(coordinates, side)

        landmarks_visible = (
            is_visible(hip, self.min_visibility)
            and is_visible(knee, self.min_visibility)
            and is_visible(ankle, self.min_visibility)
        )

        if not landmarks_visible:
            return self._status(landmarks_visible=False)

        knee_angle = calculate_angle_or_none(hip, knee, ankle)
        if knee_angle is None:
            return self._status(landmarks_visible=False)

        smoothed_angle = self._angle_smoother.update(knee_angle)

        self.current_knee_angle = smoothed_angle
        self.active_side = side
        self._update_state(smoothed_angle)

        return self._status(landmarks_visible=True)

    def _update_state(self, knee_angle: float) -> None:
        """Apply the DOWN/UP hysteresis transition described above."""
        if self.state == SquatState.UP and knee_angle < self.down_threshold:
            self.state = SquatState.DOWN

        elif self.state == SquatState.DOWN and knee_angle > self.up_threshold:
            self.state = SquatState.UP
            self.rep_count += 1

    def _status(self, landmarks_visible: bool) -> dict:
        return {
            "rep_count": self.rep_count,
            "state": self.state.value,
            "knee_angle": self.current_knee_angle,
            "side": self.active_side,
            "landmarks_visible": landmarks_visible,
        }

    def reset(self) -> None:
        """Reset the counter back to a fresh session (state UP, count 0)."""
        self.state = SquatState.UP
        self.rep_count = 0
        self.current_knee_angle = None
        self.active_side = None
        self._angle_smoother.reset()