"""
pose/landmarks.py

Purpose:
    Reusable helpers for pulling specific body landmarks out of a pose
    detection result. Centralizes landmark indices and left/right
    naming here so exercise-specific modules (squats, push-ups, curls,
    jumping jacks) never hardcode magic numbers like `coordinates[13]`
    or duplicate left/right logic.

    Everything here works off the `coordinates` dict produced by
    PoseDetector.get_landmark_coordinates() — not the raw MediaPipe
    result — so this module has no MediaPipe dependency of its own.
"""

from enum import IntEnum
from typing import NamedTuple, Optional


class Landmark(IntEnum):
    """
    Indices for the BlazePose body landmarks relevant to fitness
    tracking (a subset of the full 33-point model MediaPipe returns).
    """
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


class LandmarkPoint(NamedTuple):
    """A single landmark's pixel position and detection confidence."""
    x: int
    y: int
    visibility: float


# Named joint triplets per body side. Each maps to (near, mid, far)
# points around the joint whose angle we'll eventually measure —
# e.g. shoulder-elbow-wrist for elbow angle, hip-knee-ankle for knee
# angle. Exercise modules read from these instead of listing raw
# landmark indices themselves.
ARM_LANDMARKS = {
    "left": (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    "right": (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
}

LEG_LANDMARKS = {
    "left": (Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    "right": (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
}

HIP_LANDMARKS = {
    "left": (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    "right": (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
}


def get_landmark(coordinates: dict, landmark: Landmark) -> Optional[LandmarkPoint]:
    """
    Return the (x, y, visibility) for a single landmark from the
    coordinates dict produced by PoseDetector.get_landmark_coordinates().
    Returns None if that landmark wasn't in the dict (pose not detected,
    or MediaPipe didn't report it this frame).
    """
    entry = coordinates.get(int(landmark))
    if entry is None:
        return None
    x, y, visibility = entry
    return LandmarkPoint(x, y, visibility)


def is_visible(point: Optional[LandmarkPoint], min_visibility: float = 0.5) -> bool:
    """
    True if a landmark was detected and MediaPipe is reasonably
    confident it's genuinely visible (not occluded or off-screen).
    Exercise modules should check this before trusting a point for
    angle calculations.
    """
    return point is not None and point.visibility >= min_visibility


def get_arm_points(coordinates: dict, side: str) -> tuple:
    """
    Return (shoulder, elbow, wrist) LandmarkPoints (each possibly None)
    for the given side ("left" or "right"). Used for elbow-angle
    exercises like bicep curls and push-ups.
    """
    shoulder_lm, elbow_lm, wrist_lm = ARM_LANDMARKS[side]
    return (
        get_landmark(coordinates, shoulder_lm),
        get_landmark(coordinates, elbow_lm),
        get_landmark(coordinates, wrist_lm),
    )


def get_leg_points(coordinates: dict, side: str) -> tuple:
    """
    Return (hip, knee, ankle) LandmarkPoints for the given side. Used
    for knee-angle exercises like squats.
    """
    hip_lm, knee_lm, ankle_lm = LEG_LANDMARKS[side]
    return (
        get_landmark(coordinates, hip_lm),
        get_landmark(coordinates, knee_lm),
        get_landmark(coordinates, ankle_lm),
    )


def get_hip_points(coordinates: dict, side: str) -> tuple:
    """
    Return (shoulder, hip, knee) LandmarkPoints for the given side.
    Used for torso-angle checks, e.g. flagging a rounded back on
    squats or a sagging hip on push-ups.
    """
    shoulder_lm, hip_lm, knee_lm = HIP_LANDMARKS[side]
    return (
        get_landmark(coordinates, shoulder_lm),
        get_landmark(coordinates, hip_lm),
        get_landmark(coordinates, knee_lm),
    )


def get_midpoint(
    point_a: Optional[LandmarkPoint], point_b: Optional[LandmarkPoint]
) -> Optional[LandmarkPoint]:
    """
    Return the midpoint between two landmarks, or None if either is
    missing. Visibility of the midpoint is the lower of the two, since
    the midpoint is only as trustworthy as its weakest input.
    """
    if point_a is None or point_b is None:
        return None
    return LandmarkPoint(
        x=(point_a.x + point_b.x) // 2,
        y=(point_a.y + point_b.y) // 2,
        visibility=min(point_a.visibility, point_b.visibility),
    )


def get_shoulder_center(coordinates: dict) -> Optional[LandmarkPoint]:
    """Midpoint between both shoulders — a stable torso reference point."""
    left = get_landmark(coordinates, Landmark.LEFT_SHOULDER)
    right = get_landmark(coordinates, Landmark.RIGHT_SHOULDER)
    return get_midpoint(left, right)


def get_hip_center(coordinates: dict) -> Optional[LandmarkPoint]:
    """Midpoint between both hips — a stable torso reference point."""
    left = get_landmark(coordinates, Landmark.LEFT_HIP)
    right = get_landmark(coordinates, Landmark.RIGHT_HIP)
    return get_midpoint(left, right)


def pick_more_visible_side(coordinates: dict, joint_landmarks: dict) -> str:
    """
    Given a {"left": (...), "right": (...)} landmark triplet (e.g.
    ARM_LANDMARKS or LEG_LANDMARKS), return whichever side ("left" or
    "right") MediaPipe is more confident about, using the middle joint
    (elbow/knee) as the comparison point.

    Useful when the camera only has a clear view of one side of the
    body — a common setup for home workout videos — so exercise
    modules can automatically track whichever side is actually visible
    instead of always assuming "left".
    """
    left_middle_joint = joint_landmarks["left"][1]
    right_middle_joint = joint_landmarks["right"][1]

    left_point = get_landmark(coordinates, left_middle_joint)
    right_point = get_landmark(coordinates, right_middle_joint)

    left_visibility = left_point.visibility if left_point else 0.0
    right_visibility = right_point.visibility if right_point else 0.0

    return "left" if left_visibility >= right_visibility else "right"