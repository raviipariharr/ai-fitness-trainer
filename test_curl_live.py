"""
test_curl_live.py

Purpose:
    Tests exercises/bicep_curl.py against a real, live camera feed.
    Do a few real curls and confirm the count only increases once per
    curl. Try one deliberately "cheated" curl (swing your whole arm
    forward) and confirm the form feedback catches it.

Usage:
    python test_curl_live.py

Controls:
    Press 'q' or ESC to quit.
    Press 'r' to reset the rep count.
"""

import time

import cv2

from pose.detector import PoseDetector
from exercises.bicep_curl import BicepCurlCounter

WINDOW_NAME = "Bicep Curl Counter Test"


def draw_status(frame, status: dict) -> None:
    cv2.putText(frame, f"Curls: {status['rep_count']}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    state_color = (0, 165, 255) if status["state"] == "CONTRACTED" else (255, 255, 0)
    cv2.putText(frame, f"State: {status['state']}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    if status["elbow_angle"] is not None:
        cv2.putText(frame, f"Elbow ({status['side']}): {status['elbow_angle']:.0f} deg", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Arm not clearly visible", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if status["last_rep_feedback"]:
        color = (0, 200, 0) if status["last_rep_feedback"] == "Good rep" else (0, 0, 255)
        cv2.putText(frame, status["last_rep_feedback"], (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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

    curl_counter = BicepCurlCounter()
    print("Curl test started. Press 'q' to quit, 'r' to reset.")
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

            status = curl_counter.update(coordinates) if coordinates else {
                "rep_count": curl_counter.rep_count, "state": curl_counter.state.value,
                "elbow_angle": None, "side": None, "last_rep_feedback": curl_counter.last_rep_feedback,
            }
            draw_status(frame, status)

            cv2.imshow(WINDOW_NAME, frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key_pressed in (ord("q"), 27) or window_closed:
                break
            if key_pressed == ord("r"):
                curl_counter.reset()
                print("Rep count reset.")
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()
        print(f"Test finished. Final curl count: {curl_counter.rep_count}")


if __name__ == "__main__":
    main()