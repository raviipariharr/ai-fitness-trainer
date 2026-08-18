"""
exercises/pushup.py

Purpose:
    Detects push-ups and counts repetitions using the elbow angle
    (shoulder-elbow-wrist) — the same UP/DOWN hysteresis state machine
    as squats (see exercises/squat.py for why two thresholds prevent
    double counting), just applied to the arm instead of the leg.

    Also gives two basic, real-world form checks:
      1. Body alignment (shoulder-hip-ankle angle) — catches the most
         common push-up fault: hips sagging toward the floor or piking
         up into the air instead of keeping a straight plank line.
      2. Rep depth — flags a push-up that never got the elbow low
         enough to count as a real full-range rep.

Why elbow angle:
    Top of a push-up: arms extended, angle near 180 deg.
    Bottom of a push-up: arms bent, angle near 90 deg or lower.
    That's a wide, reliable swing to detect state from — same reasoning
    as knee angle for squats.
"""

from enum import Enum
from typing import Optional

from pose.landmarks import (
    ARM_LANDMARKS,
    Landmark,
    get_arm_points,
    get_landmark,
    is_visible,
    pick_more_visible_side,
)
from utils.angles import calculate_angle_or_none


class PushupState(Enum):
    UP = "UP"
    DOWN = "DOWN"


class PushupCounter:
    """
    Stateful push-up rep counter with basic form feedback. Create one
    instance per workout session and call update() once per frame with
    the current landmark coordinates.
    """

    def __init__(
        self,
        down_threshold: float = 90.0,
        up_threshold: float = 160.0,
        good_depth_threshold: float = 100.0,
        straight_body_threshold: float = 160.0,
        min_visibility: float = 0.5,
    ):
        if down_threshold >= up_threshold:
            raise ValueError("down_threshold must be lower than up_threshold")

        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.good_depth_threshold = good_depth_threshold
        self.straight_body_threshold = straight_body_threshold
        self.min_visibility = min_visibility

        self.state = PushupState.UP
        self.rep_count = 0
        self.current_elbow_angle: Optional[float] = None
        self.active_side: Optional[str] = None
        self._min_angle_this_rep: Optional[float] = None
        self.last_rep_feedback: Optional[str] = None

    def update(self, coordinates: dict) -> dict:
        """
        Process one frame's landmark coordinates. Returns a status
        dict:
            {
                "rep_count": int,
                "state": "UP" | "DOWN",
                "elbow_angle": float | None,
                "side": "left" | "right" | None,
                "landmarks_visible": bool,
                "body_feedback": str | None,   # live, every frame
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
            return self._status(landmarks_visible=False, body_feedback=None)

        elbow_angle = calculate_angle_or_none(shoulder, elbow, wrist)
        if elbow_angle is None:
            return self._status(landmarks_visible=False, body_feedback=None)

        self.current_elbow_angle = elbow_angle
        self.active_side = side
        self._update_state(elbow_angle)

        body_feedback = self._check_body_alignment(coordinates, side, shoulder)

        return self._status(landmarks_visible=True, body_feedback=body_feedback)

    def _update_state(self, elbow_angle: float) -> None:
        """Same DOWN/UP hysteresis pattern as SquatCounter — see squat.py."""
        if self.state == PushupState.UP and elbow_angle < self.down_threshold:
            self.state = PushupState.DOWN
            self._min_angle_this_rep = elbow_angle

        elif self.state == PushupState.DOWN:
            if self._min_angle_this_rep is None or elbow_angle < self._min_angle_this_rep:
                self._min_angle_this_rep = elbow_angle

            if elbow_angle > self.up_threshold:
                self.state = PushupState.UP
                self.rep_count += 1

                if self._min_angle_this_rep is not None and self._min_angle_this_rep > self.good_depth_threshold:
                    self.last_rep_feedback = "Partial rep - lower your chest further next time"
                else:
                    self.last_rep_feedback = "Good depth"

                self._min_angle_this_rep = None

    def _check_body_alignment(self, coordinates: dict, side: str, shoulder) -> Optional[str]:
        """
        Check the shoulder-hip-ankle angle. A straight plank line reads
        close to 180 degrees; a sagging or piked hip pulls it down.
        Returns a feedback message, or None if alignment looks fine or
        the hip/ankle aren't clearly visible.
        """
        hip_landmark = Landmark.LEFT_HIP if side == "left" else Landmark.RIGHT_HIP
        ankle_landmark = Landmark.LEFT_ANKLE if side == "left" else Landmark.RIGHT_ANKLE

        hip = get_landmark(coordinates, hip_landmark)
        ankle = get_landmark(coordinates, ankle_landmark)

        if not is_visible(hip, self.min_visibility) or not is_visible(ankle, self.min_visibility):
            return None

        body_line_angle = calculate_angle_or_none(shoulder, hip, ankle)
        if body_line_angle is not None and body_line_angle < self.straight_body_threshold:
            return "Keep your body straight - engage your core"

        return None

    def _status(self, landmarks_visible: bool, body_feedback: Optional[str]) -> dict:
        return {
            "rep_count": self.rep_count,
            "state": self.state.value,
            "elbow_angle": self.current_elbow_angle,
            "side": self.active_side,
            "landmarks_visible": landmarks_visible,
            "body_feedback": body_feedback,
            "last_rep_feedback": self.last_rep_feedback,
        }

    def reset(self) -> None:
        """Reset the counter back to a fresh session (state UP, count 0)."""
        self.state = PushupState.UP
        self.rep_count = 0
        self.current_elbow_angle = None
        self.active_side = None
        self._min_angle_this_rep = None
        self.last_rep_feedback = None