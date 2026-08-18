"""
test_jack_live.py

Purpose:
    Tests exercises/jumping_jack.py against a real, live camera feed.
    Stand back far enough that your whole body (both wrists, both
    ankles) fits in frame, do a few real jumping jacks, and confirm
    the count increments once per jack. Also try raising just your
    arms without jumping, and jumping your feet apart without raising
    your arms — neither should count.

Usage:
    python test_jack_live.py

Controls:
    Press 'q' or ESC to quit.
    Press 'r' to reset the rep count.
"""

import time

import cv2

from pose.detector import PoseDetector
from exercises.jumping_jack import JumpingJackCounter

WINDOW_NAME = "Jumping Jack Counter Test"


def draw_status(frame, status: dict) -> None:
    cv2.putText(frame, f"Jacks: {status['rep_count']}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    state_color = (0, 165, 255) if status["state"] == "OPEN" else (255, 255, 0)
    cv2.putText(frame, f"State: {status['state']}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    if status["arm_raise_ratio"] is not None and status["leg_spread_ratio"] is not None:
        detail_text = f"arm={status['arm_raise_ratio']:.2f}  legs={status['leg_spread_ratio']:.2f}"
        cv2.putText(frame, detail_text, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Full body not clearly visible", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


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

    jack_counter = JumpingJackCounter()
    print("Jumping jack test started. Stand back so your whole body is in frame.")
    print("Press 'q' to quit, 'r' to reset.")
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

            status = jack_counter.update(coordinates) if coordinates else {
                "rep_count": jack_counter.rep_count, "state": jack_counter.state.value,
                "arm_raise_ratio": None, "leg_spread_ratio": None,
            }
            draw_status(frame, status)

            cv2.imshow(WINDOW_NAME, frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key_pressed in (ord("q"), 27) or window_closed:
                break
            if key_pressed == ord("r"):
                jack_counter.reset()
                print("Rep count reset.")
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()
        print(f"Test finished. Final jumping jack count: {jack_counter.rep_count}")


if __name__ == "__main__":
    main()