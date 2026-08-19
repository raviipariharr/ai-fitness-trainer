"""
exercises/base.py

Purpose:
    Defines the one interface every exercise tracker implements, so
    main.py can drive whichever exercise is active through the exact
    same handful of calls without ever knowing which exercise it
    actually is. This is what makes exercise selection a dictionary
    lookup instead of an if/elif chain: main.py never asks "is this a
    squat or a push-up?" — it just calls tracker.update(coordinates)
    and reads a standardized status dict back, whatever's underneath.
"""

from abc import ABC, abstractmethod


class ExerciseTracker(ABC):
    """
    Common interface wrapping an exercise-specific counter (SquatCounter,
    PushupCounter, BicepCurlCounter, JumpingJackCounter). Each concrete
    subclass in exercises/registry.py adapts one counter's particular
    status dict into this shared shape.
    """

    display_name: str = "Exercise"

    @abstractmethod
    def update(self, coordinates: dict) -> dict:
        """
        Process one frame's landmark coordinates. Must return a dict
        with exactly these keys, regardless of which exercise this is:
            {
                "rep_count": int,
                "state": str,               # exercise-specific state name
                "feedback": str | None,      # form feedback, if any
                "detail": str | None,        # e.g. "left knee: 92 deg"
                "landmarks_visible": bool,
            }
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the rep count and internal state back to a fresh session."""
        raise NotImplementedError