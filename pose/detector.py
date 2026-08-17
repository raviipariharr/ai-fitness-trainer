"""
pose/detector.py

Purpose:
    Wraps MediaPipe's PoseLandmarker (the current Tasks API) so the rest
    of the application never touches MediaPipe setup directly. Feed it a
    BGR frame from OpenCV; it hands back landmark positions and can draw
    the skeleton for visual feedback.

    We use the Tasks API (not the old mp.solutions.pose) because the
    legacy Solutions API is deprecated and broken on current MediaPipe
    releases (see Phase 1 notes in the README).
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "pose_landmarker_lite.task"
)

# The 33-landmark BlazePose skeleton connections. Same landmark index
# layout is used by both the legacy and current MediaPipe pose models,
# so this list is stable regardless of which API version is running.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

LANDMARK_NAMES = {
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}


class PoseDetector:
    """
    Thin, synchronous wrapper around MediaPipe's PoseLandmarker for
    frame-by-frame video processing (one call per webcam frame).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found at '{model_path}'.\n"
                "Run 'python download_model.py' first to download it."
            )

        base_options = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms: int):
        """
        Run pose detection on one BGR frame. timestamp_ms must increase
        monotonically between calls (required by VIDEO running mode —
        e.g. pass elapsed milliseconds since the program started).

        Returns a PoseLandmarkerResult. result.pose_landmarks is a list
        of detected poses; each pose is a list of 33 normalized
        landmarks (x, y, z in [0, 1], plus visibility/presence). Empty
        list if no person was detected in this frame.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame_bgr, detection_result) -> None:
        """
        Draw the detected skeleton directly onto frame_bgr, in place.
        Does nothing if no pose was detected in this frame.
        """
        if not detection_result.pose_landmarks:
            return

        height, width = frame_bgr.shape[:2]
        landmarks = detection_result.pose_landmarks[0]  # first person only
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(frame_bgr, points[start_idx], points[end_idx], (0, 255, 0), 2)

        for point in points:
            cv2.circle(frame_bgr, point, 4, (0, 0, 255), -1)

    def get_landmark_coordinates(self, detection_result, frame_shape) -> dict:
        """
        Return {landmark_index: (x_pixels, y_pixels, visibility)} for the
        first detected pose, or {} if nothing was detected. This is what
        Phase 4's joint-angle math will read from.
        """
        if not detection_result.pose_landmarks:
            return {}

        height, width = frame_shape[:2]
        landmarks = detection_result.pose_landmarks[0]

        return {
            index: (int(lm.x * width), int(lm.y * height), round(lm.visibility, 2))
            for index, lm in enumerate(landmarks)
        }

    def close(self) -> None:
        """Release the underlying MediaPipe landmarker."""
        self._landmarker.close()