"""
main.py

Purpose:
    Entry point for the AI Fitness Trainer. In this phase, it only opens
    the webcam and displays the live video feed in a window. No pose
    detection, exercise logic, or UI overlays yet — this just proves the
    video pipeline works before we build anything on top of it.

Usage:
    python main.py

Controls:
    Press 'q' or ESC to quit.
    Closing the window (clicking X) also exits cleanly.
"""

import cv2

WINDOW_NAME = "AI Fitness Trainer"
WEBCAM_INDEX = 0


def open_webcam(index: int) -> cv2.VideoCapture:
    """
    Open the webcam at the given index and return the capture object.
    Raises a RuntimeError with a clear message if the webcam can't be
    opened, instead of letting the program crash with a cryptic OpenCV
    error later when we try to read a frame.
    """
    capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {index}. "
            "Check that a camera is connected, no other app is using it, "
            "and your OS has granted camera permission to this program."
        )

    return capture


def run_video_loop(capture: cv2.VideoCapture) -> None:
    """
    Continuously read frames from the webcam and display them in a
    window until the user quits. Handles the case where a frame read
    fails mid-stream (e.g. camera disconnected) without crashing.
    """
    print("Webcam feed started. Press 'q' or ESC to quit.")

    while True:
        frame_read_successfully, frame = capture.read()

        if not frame_read_successfully:
            print("Warning: failed to read a frame from the webcam. Stopping.")
            break

        cv2.imshow(WINDOW_NAME, frame)

        key_pressed = cv2.waitKey(1) & 0xFF
        quit_requested = key_pressed in (ord("q"), 27)  # 27 = ESC
        window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1

        if quit_requested or window_closed:
            break


def main() -> None:
    try:
        capture = open_webcam(WEBCAM_INDEX)
    except RuntimeError as error:
        print(f"Error: {error}")
        return

    try:
        run_video_loop(capture)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Webcam released. Exited cleanly.")


if __name__ == "__main__":
    main()