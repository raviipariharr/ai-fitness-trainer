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


class SquatTracker(ExerciseTracker):
    display_name = "Squats"

    def __init__(self):
        self._counter = SquatCounter()

    def update(self, coordinates: dict) -> dict:
        status = self._counter.update(coordinates)
        angle = status["knee_angle"]
        detail = f"{status['side']} knee: {angle:.0f} deg" if angle is not None else None
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
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
        # Live body-alignment warnings take priority over the last
        # completed rep's feedback, since posture matters every frame.
        feedback = status["body_feedback"] or status["last_rep_feedback"]
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
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
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
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
        return {
            "rep_count": status["rep_count"],
            "state": status["state"],
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