"""
test_squat_live.py

Purpose:
    Tests exercises/squat.py against a real, live camera feed. Runs
    pose detection, feeds landmark coordinates into SquatCounter each
    frame, and displays the rep count, current state (UP/DOWN), and
    live knee angle on screen.

    Do a few real squats in front of the camera and confirm the count
    only goes up once per squat — not per frame, not per wobble.

Usage:
    python test_squat_live.py

Controls:
    Press 'q' or ESC to quit.
    Press 'r' to reset the rep count to zero.
"""

import time

import cv2

from pose.detector import PoseDetector
from exercises.squat import SquatCounter

WINDOW_NAME = "Squat Counter Test"


def draw_status(frame, status: dict) -> None:
    """Draw rep count, state, and knee angle in the top-left corner."""
    rep_text = f"Squats: {status['rep_count']}"
    cv2.putText(frame, rep_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    state_color = (0, 165, 255) if status["state"] == "DOWN" else (255, 255, 0)
    cv2.putText(frame, f"State: {status['state']}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    if status["knee_angle"] is not None:
        angle_text = f"Knee angle ({status['side']}): {status['knee_angle']:.0f} deg"
        cv2.putText(frame, angle_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Leg not clearly visible", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


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

    squat_counter = SquatCounter()
    print("Squat test started. Do some squats. Press 'q' to quit, 'r' to reset count.")
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

            status = squat_counter.update(coordinates) if coordinates else {
                "rep_count": squat_counter.rep_count, "state": squat_counter.state.value,
                "knee_angle": None, "side": None,
            }
            draw_status(frame, status)

            cv2.imshow(WINDOW_NAME, frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key_pressed in (ord("q"), 27) or window_closed:
                break
            if key_pressed == ord("r"):
                squat_counter.reset()
                print("Rep count reset.")
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()
        print(f"Test finished. Final squat count: {squat_counter.rep_count}")


if __name__ == "__main__":
    main()