"""
utils/angles.py

Purpose:
    One reusable function for calculating the angle at a joint, given
    three points: the joint itself (the "vertex") and the two points on
    either side of it. This is the core math every exercise module
    builds on — a squat is really just "watch the hip-knee-ankle angle
    go from ~170 degrees to under 90 and back."

The math, explained:
    Say we want the angle AT the elbow, formed by the upper arm
    (shoulder -> elbow) and forearm (elbow -> wrist).

    1. Treat the elbow as the vertex "B", the shoulder as "A", and the
       wrist as "C".

    2. Build two vectors that both start at the vertex and point
       outward to the other two landmarks:
           BA = A - B   (vector from elbow to shoulder)
           BC = C - B   (vector from elbow to wrist)

       We build vectors from the vertex outward (not A->C directly)
       because the angle we want is the angle BETWEEN these two arms
       of the joint, and the dot product formula below only works on
       vectors that share a common starting point.

    3. The dot product of two vectors relates to the angle between
       them by definition:
           BA . BC = |BA| * |BC| * cos(theta)

       Rearranging for theta:
           cos(theta) = (BA . BC) / (|BA| * |BC|)

       where |BA| and |BC| are the vector lengths (magnitudes), and
       "." is the dot product: (ba_x * bc_x) + (ba_y * bc_y).

    4. Take arccos (inverse cosine) of that ratio to get theta in
       radians, then convert to degrees. The result is always between
       0 and 180 degrees, which is exactly the range a real joint angle
       falls in (0 = fully folded, 180 = fully straight).

    5. Floating-point rounding can occasionally push the cosine value
       a hair outside [-1, 1] (e.g. 1.0000000002), which would make
       arccos return NaN. We clip it back into range before taking
       arccos.

    This is a 2D calculation (x, y pixel coordinates only, ignoring
    MediaPipe's z depth estimate). Depth estimates from a single webcam
    are noisy, and 2D angles are accurate enough for judging exercise
    form from a roughly front-on or side-on camera angle.
"""

import math
from typing import Optional, Tuple, Union

Point = Union[Tuple[float, float], "LandmarkPointLike"]


def _extract_xy(point) -> Tuple[float, float]:
    """
    Accept either a LandmarkPoint-style object (anything with .x and
    .y attributes, from pose/landmarks.py) or a plain (x, y) tuple, so
    this function stays reusable outside the pose-detection pipeline
    too — e.g. for testing with hand-picked coordinates.
    """
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def calculate_angle(point_a, point_b, point_c) -> float:
    """
    Calculate the angle in degrees at point_b (the vertex), formed by
    the segments point_b->point_a and point_b->point_c.

    Example: calculate_angle(shoulder, elbow, wrist) returns the elbow
    angle. calculate_angle(hip, knee, ankle) returns the knee angle.

    Returns a value in [0, 180]. Raises ValueError if any point is None
    (use calculate_angle_or_none for a safe version that handles that).
    """
    if point_a is None or point_b is None or point_c is None:
        raise ValueError("calculate_angle received a None point — check visibility first.")

    ax, ay = _extract_xy(point_a)
    bx, by = _extract_xy(point_b)
    cx, cy = _extract_xy(point_c)

    vector_ba = (ax - bx, ay - by)
    vector_bc = (cx - bx, cy - by)

    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]
    magnitude_ba = math.hypot(*vector_ba)
    magnitude_bc = math.hypot(*vector_bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        raise ValueError("calculate_angle received two identical points — angle is undefined.")

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))  # guard against float drift outside [-1, 1]

    angle_radians = math.acos(cosine_angle)
    return math.degrees(angle_radians)


def calculate_angle_or_none(point_a, point_b, point_c) -> Optional[float]:
    """
    Safe wrapper around calculate_angle: returns None instead of
    raising if any point is None or invalid. This is what exercise
    modules should actually call, since pose/landmarks.py points are
    frequently None (a joint wasn't detected this frame).
    """
    try:
        return calculate_angle(point_a, point_b, point_c)
    except ValueError:
        return None