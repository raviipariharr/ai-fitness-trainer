"""
utils/smoothing.py

Purpose:
    A single reusable smoothing filter, applied to noisy angle/ratio
    readings before they reach an exercise's state machine. This is
    not just cosmetic — it fixes a real bug: a single bad MediaPipe
    frame (self-occlusion, motion blur, a momentary misdetection) can
    swing a raw angle reading far enough to flip a state and register
    a phantom rep, even though the person never actually moved. See
    the smoothing test suite for a concrete reproduction of this.

Why exponential moving average (EMA), not a simple moving average:
    A simple moving average needs a full window of readings before it
    means anything, and it lags behind real movement by roughly half
    the window size — noticeable as sluggish response in a real-time
    app. An EMA has no warm-up period (it starts from the very first
    reading) and reacts immediately, just proportionally damped:

        smoothed = alpha * new_reading + (1 - alpha) * smoothed_previous

    `alpha` controls the trade-off: closer to 1.0 means "trust the new
    reading almost entirely" (fast, less smoothing); closer to 0.0
    means "barely move from the previous smoothed value" (slow, more
    smoothing). alpha=1.0 is equivalent to no smoothing at all.
"""

from typing import Optional


class ExponentialMovingAverage:
    """
    Smooths a stream of numeric readings. Call update() once per
    frame with the latest raw reading (or None if this frame's
    reading is unavailable) and use the returned value in place of
    the raw one everywhere downstream — state machine comparisons,
    on-screen display, everything.
    """

    def __init__(self, alpha: float = 0.4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0.0, 1.0]")
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_reading: Optional[float]) -> Optional[float]:
        """
        Feed in the latest raw reading. Returns the updated smoothed
        value. If new_reading is None (nothing detected this frame),
        the smoothed value is left untouched and returned as-is —
        holding the last known value is more honest than guessing, and
        matches how every exercise counter already treats a missing
        landmark.
        """
        if new_reading is None:
            return self.value

        if self.value is None:
            # First real reading — nothing to blend with yet, so start
            # exactly there instead of smoothing toward a made-up prior.
            self.value = new_reading
        else:
            self.value = self.alpha * new_reading + (1 - self.alpha) * self.value

        return self.value

    def reset(self) -> None:
        """Clear all history, e.g. when an exercise counter is reset."""
        self.value = None