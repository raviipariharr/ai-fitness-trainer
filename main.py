"""
main.py

Purpose:
    Entry point for the AI Fitness Trainer. Opens the webcam, runs
    pose detection on each frame, and drives whichever exercise the
    user has selected through the ExerciseTracker interface
    (exercises/base.py). Reps, state, and form feedback all come from
    the tracker's standardized status dict.

    This file has no idea what a squat or a push-up is. Exercise
    selection is a single dict lookup into EXERCISE_REGISTRY
    (exercises/registry.py) — there is no if/elif chain checking which
    exercise is active anywhere in this file. Adding a fifth exercise
    means adding one entry to the registry and one line to
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


def draw_menu(frame) -> None:
    """
    List every available exercise and its selection key along the
    bottom of the frame. Built entirely from SELECTION_KEYS and the
    registry's display_name - adding an exercise updates this menu
    automatically, no code change needed here.
    """
    height = frame.shape[0]
    y = height - 15 - (len(SELECTION_KEYS) * 20)

    for key_code, registry_key in SELECTION_KEYS.items():
        tracker_class = EXERCISE_REGISTRY[registry_key]
        label = f"[{chr(key_code)}] {tracker_class.display_name}"
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 20


def draw_status(frame, tracker, status: dict) -> None:
    """
    Draw exercise name, rep count, state, and any feedback/detail the
    tracker provided. Reads only the standardized ExerciseTracker
    status shape, so this works identically for every exercise.
    """
    cv2.putText(frame, tracker.display_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Reps: {status['rep_count']}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(frame, f"State: {status['state']}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y = 120
    if not status["landmarks_visible"]:
        cv2.putText(frame, "Body not clearly visible", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        y += 22
    elif status["detail"]:
        cv2.putText(frame, status["detail"], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22

    if status["feedback"]:
        cv2.putText(frame, status["feedback"], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)


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
        draw_status(frame, tracker, status)
        draw_menu(frame)

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