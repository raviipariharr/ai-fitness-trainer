"""
exercises/registry.py

Purpose:
    This is the ONLY file in the whole project that knows all four
    exercises exist at once. Everywhere else — main.py included —
    works through the ExerciseTracker interface (exercises/base.py)
    and never branches on which exercise is active.

    Two things live here:
      1. Four small adapter classes, one per exercise, each wrapping
         its counter (SquatCounter, PushupCounter, etc.) and
         translating that counter's particular status dict into the
         shared ExerciseTracker shape. The counters themselves stay
         completely unchanged and unaware this layer exists — this is
         what "keep exercise-specific logic isolated" means: squat.py
         still only knows about squats.
      2. EXERCISE_REGISTRY, a plain dict mapping a short key (what the
         user selects) to its adapter class. Selecting an exercise is
         a dictionary lookup: EXERCISE_REGISTRY[key](). Adding a fifth
         exercise later means writing one more adapter class and
         adding one line to this dict — nothing in main.py changes.
"""

from .base import ExerciseTracker
from .bicep_curl import BicepCurlCounter
from .jumping_jack import JumpingJackCounter
from .pushup import PushupCounter
from .squat import SquatCounter


def _normalize_progress(value: float, resting_value: float, peak_value: float) -> float:
    """
    Map a raw angle or ratio onto a 0.0-1.0 progress scale, where 0.0
    is the resting position (e.g. standing, arm extended) and 1.0 is
    the peak of the movement (e.g. squat depth, full curl). Works
    whether the peak value is numerically lower than resting (angles,
    which shrink as a joint bends) or higher (jumping jack ratios,
    which grow as arms/legs open) — the direction is inferred from
    which of resting_value/peak_value is larger. Clipped to [0, 1]
    since live sensor readings can briefly overshoot either end.
    """
    if resting_value == peak_value:
        return 0.0
    raw_progress = (value - resting_value) / (peak_value - resting_value)
    return max(0.0, min(1.0, raw_progress))


class SquatTracker(ExerciseTracker):
    display_name = "Squats"

    def __init__(self):
        self._counter = SquatCounter()

    def update(self, coordinates: dict) -> dict:
        status = self._counter.update(coordinates)
        angle = status["knee_angle"]
        detail = f"{status['side']} knee: {angle:.0f} deg" if angle is not None else None
        progress = (
            _normalize_progress(angle, self._counter.up_threshold, self._counter.down_threshold)
            if angle is not None
            else 0.0
        )
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
            "progress": progress,
            "feedback": None,
            "detail": detail,
            "landmarks_visible": status["landmarks_visible"],
        }

    def reset(self) -> None:
        self._counter.reset()


class PushupTracker(ExerciseTracker):
    display_name = "Push-ups"

    def __init__(self):
        self._counter = PushupCounter()

    def update(self, coordinates: dict) -> dict:
        status = self._counter.update(coordinates)
        angle = status["elbow_angle"]
        detail = f"{status['side']} elbow: {angle:.0f} deg" if angle is not None else None
        progress = (
            _normalize_progress(angle, self._counter.up_threshold, self._counter.down_threshold)
            if angle is not None
            else 0.0
        )
        # Live body-alignment warnings take priority over the last
        # completed rep's feedback, since posture matters every frame.
        feedback = status["body_feedback"] or status["last_rep_feedback"]
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
            "progress": progress,
            "feedback": feedback,
            "detail": detail,
            "landmarks_visible": status["landmarks_visible"],
        }

    def reset(self) -> None:
        self._counter.reset()


class BicepCurlTracker(ExerciseTracker):
    display_name = "Bicep Curls"

    def __init__(self):
        self._counter = BicepCurlCounter()

    def update(self, coordinates: dict) -> dict:
        status = self._counter.update(coordinates)
        angle = status["elbow_angle"]
        detail = f"{status['side']} elbow: {angle:.0f} deg" if angle is not None else None
        progress = (
            _normalize_progress(angle, self._counter.extended_threshold, self._counter.contracted_threshold)
            if angle is not None
            else 0.0
        )
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
            "progress": progress,
            "feedback": status["last_rep_feedback"],
            "detail": detail,
            "landmarks_visible": status["landmarks_visible"],
        }

    def reset(self) -> None:
        self._counter.reset()


class JumpingJackTracker(ExerciseTracker):
    display_name = "Jumping Jacks"

    def __init__(self):
        self._counter = JumpingJackCounter()

    def update(self, coordinates: dict) -> dict:
        status = self._counter.update(coordinates)
        arm_ratio = status["arm_raise_ratio"]
        leg_ratio = status["leg_spread_ratio"]
        detail = (
            f"arm: {arm_ratio:.2f}  legs: {leg_ratio:.2f}"
            if arm_ratio is not None and leg_ratio is not None
            else None
        )
        if arm_ratio is not None and leg_ratio is not None:
            arm_progress = _normalize_progress(
                arm_ratio, self._counter.arm_lowered_ratio, self._counter.arm_raised_ratio
            )
            leg_progress = _normalize_progress(
                leg_ratio, self._counter.legs_closed_ratio, self._counter.legs_open_ratio
            )
            # Average the two — a jumping jack only really completes
            # when arms and legs are both partway there together, so a
            # single combined bar tells a more honest story than
            # showing whichever limb happens to be further along.
            progress = (arm_progress + leg_progress) / 2
        else:
            progress = 0.0
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
            "progress": progress,
            "feedback": None,
            "detail": detail,
            "landmarks_visible": status["landmarks_visible"],
        }

    def reset(self) -> None:
        self._counter.reset()


# The single dispatch table exercise selection is built on. Keys are
# what the user types/selects; values are the adapter class to
# instantiate. main.py only ever does EXERCISE_REGISTRY[key]() — no
# per-exercise branching anywhere outside this file.
EXERCISE_REGISTRY = {
    "squat": SquatTracker,
    "pushup": PushupTracker,
    "curl": BicepCurlTracker,
    "jumping_jack": JumpingJackTracker,
}