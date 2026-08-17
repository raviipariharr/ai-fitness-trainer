"""
test_angles_live.py

Purpose:
    Tests utils/angles.py against a real, live camera feed rather than
    hand-picked coordinates. Detects your pose, picks whichever side of
    your body the camera sees best, and overlays the live elbow angle
    (shoulder-elbow-wrist) and knee angle (hip-knee-ankle) on screen.

    Move your arm and leg while this runs — the numbers should track
    smoothly: near 180 when straight, dropping as you bend the joint.

Usage:
    python test_angles_live.py

Controls:
    Press 'q' or ESC to quit.
"""

import time

import cv2

from pose.detector import PoseDetector
from pose.landmarks import ARM_LANDMARKS, LEG_LANDMARKS, get_arm_points, get_leg_points, pick_more_visible_side
from utils.angles import calculate_angle_or_none

WINDOW_NAME = "Angle Calculation Test"


def draw_angle_label(frame, angle: float, joint_point, label: str) -> None:
    """Draw the numeric angle next to the joint it was measured at."""
    if angle is None or joint_point is None:
        return
    text = f"{label}: {angle:.0f} deg"
    position = (joint_point.x + 10, joint_point.y)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def main() -> None:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: could not open webcam.")
        return

    try:
        detector = PoseDetector()
    except FileNotFoundError as error:
        print(f"Error: {error}")
        capture.release()
        return

    print("Angle test started. Move your arm and leg. Press 'q' or ESC to quit.")
    start_time = time.time()

    try:
        while True:
            frame_read_successfully, frame = capture.read()
            if not frame_read_successfully:
                print("Warning: failed to read a frame. Stopping.")
                break

            timestamp_ms = int((time.time() - start_time) * 1000)
            result = detector.detect(frame, timestamp_ms)
            detector.draw_landmarks(frame, result)
            coordinates = detector.get_landmark_coordinates(result, frame.shape)

            if coordinates:
                arm_side = pick_more_visible_side(coordinates, ARM_LANDMARKS)
                leg_side = pick_more_visible_side(coordinates, LEG_LANDMARKS)

                shoulder, elbow, wrist = get_arm_points(coordinates, arm_side)
                elbow_angle = calculate_angle_or_none(shoulder, elbow, wrist)
                draw_angle_label(frame, elbow_angle, elbow, f"{arm_side} elbow")

                hip, knee, ankle = get_leg_points(coordinates, leg_side)
                knee_angle = calculate_angle_or_none(hip, knee, ankle)
                draw_angle_label(frame, knee_angle, knee, f"{leg_side} knee")

            cv2.imshow(WINDOW_NAME, frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key_pressed in (ord("q"), 27) or window_closed:
                break
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()
        print("Test finished. Webcam released.")


if __name__ == "__main__":
    main()