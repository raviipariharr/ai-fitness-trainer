"""
main.py

Purpose:
    Entry point for the AI Fitness Trainer. Opens the webcam, runs
    pose detection on each frame, and drives whichever exercise the
    user has selected through the ExerciseTracker interface
    (exercises/base.py). Reps, state, progress, and form feedback all
    come from the tracker's standardized status dict; the on-screen
    look of all that is handled by ui/overlay.py, not by this file.

    This file has no idea what a squat or a push-up is, and no idea
    what a progress bar looks like. Exercise selection is a single
    dict lookup into EXERCISE_REGISTRY (exercises/registry.py); the
    HUD is a single call to ui.overlay.render_hud(). Adding a fifth
    exercise means adding one entry to the registry and one line to
    SELECTION_KEYS below; nothing else here changes.

Usage:
    python main.py
    (Run download_model.py once first if you haven't already.)

Controls:
    1 - Squats
    2 - Push-ups
    3 - Bicep Curls
    4 - Jumping Jacks
    r - Reset the current exercise's rep count
    q / ESC - Quit
"""

import time

import cv2

from pose.detector import PoseDetector
from exercises.registry import EXERCISE_REGISTRY
from ui.overlay import render_hud

WINDOW_NAME = "AI Fitness Trainer"
WEBCAM_INDEX = 0

# Maps a keypress to a registry key. This is the only place keyboard
# keys and exercise names meet - registry.py doesn't know about
# keyboard input, and this dict doesn't know how any exercise works.
SELECTION_KEYS = {
    ord("1"): "squat",
    ord("2"): "pushup",
    ord("3"): "curl",
    ord("4"): "jumping_jack",
}
DEFAULT_EXERCISE_KEY = "squat"


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


def run_video_loop(capture: cv2.VideoCapture, detector: PoseDetector) -> None:
    """
    Continuously read frames, run pose detection, feed landmarks to
    whichever exercise tracker is active, and display the result.
    Handles a failed frame read (e.g. camera disconnected) without
    crashing.
    """
    current_key = DEFAULT_EXERCISE_KEY
    tracker = EXERCISE_REGISTRY[current_key]()

    print(f"Webcam feed started. Tracking: {tracker.display_name}.")
    print("Press 1-4 to switch exercise, 'r' to reset, 'q'/ESC to quit.")
    start_time = time.time()

    while True:
        frame_read_successfully, frame = capture.read()

        if not frame_read_successfully:
            print("Warning: failed to read a frame from the webcam. Stopping.")
            break

        timestamp_ms = int((time.time() - start_time) * 1000)
        detection_result = detector.detect(frame, timestamp_ms)
        detector.draw_landmarks(frame, detection_result)
        coordinates = detector.get_landmark_coordinates(detection_result, frame.shape)

        status = tracker.update(coordinates)
        render_hud(frame, tracker.display_name, status, EXERCISE_REGISTRY, SELECTION_KEYS, current_key)

        cv2.imshow(WINDOW_NAME, frame)

        key_pressed = cv2.waitKey(1) & 0xFF
        quit_requested = key_pressed in (ord("q"), 27)  # 27 = ESC
        window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1

        if quit_requested or window_closed:
            break

        if key_pressed in SELECTION_KEYS and SELECTION_KEYS[key_pressed] != current_key:
            current_key = SELECTION_KEYS[key_pressed]
            tracker = EXERCISE_REGISTRY[current_key]()
            print(f"Switched to: {tracker.display_name}")
        elif key_pressed == ord("r"):
            tracker.reset()
            print(f"{tracker.display_name} count reset.")


def main() -> None:
    try:
        capture = open_webcam(WEBCAM_INDEX)
    except RuntimeError as error:
        print(f"Error: {error}")
        return

    try:
        detector = PoseDetector()
    except FileNotFoundError as error:
        print(f"Error: {error}")
        capture.release()
        return

    try:
        run_video_loop(capture, detector)
    finally:
        detector.close()
        capture.release()
        cv2.destroyAllWindows()
        print("Webcam released. Exited cleanly.")


if __name__ == "__main__":
    main()