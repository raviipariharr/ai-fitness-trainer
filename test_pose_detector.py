"""
test_pose_detector.py

Purpose:
    Tests the pose/detector.py module on its own, independent of
    main.py's full pipeline. Opens the webcam, runs PoseDetector on
    each frame, draws the skeleton, and prints a couple of key landmark
    coordinates to the console so you can visually confirm detection is
    accurate before wiring it into the rest of the app.

Usage:
    python test_pose_detector.py

Controls:
    Press 'q' or ESC to quit.
"""

import time

import cv2

from pose.detector import PoseDetector, LANDMARK_NAMES

WINDOW_NAME = "Pose Detection Test"


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

    start_time = time.time()
    print("Pose detection test started. Press 'q' or ESC to quit.")

    frame_count = 0
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

            # Print a couple of key joints to the console every 30 frames
            # (roughly once a second) so the terminal isn't flooded.
            frame_count += 1
            if coordinates and frame_count % 30 == 0:
                for index, name in LANDMARK_NAMES.items():
                    if index in coordinates:
                        x, y, visibility = coordinates[index]
                        print(f"{name}: x={x}, y={y}, visibility={visibility}")
                print("-" * 40)

            status_text = "Pose detected" if coordinates else "No pose detected"
            cv2.putText(
                frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

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