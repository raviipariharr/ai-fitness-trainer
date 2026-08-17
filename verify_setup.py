"""
verify_setup.py

Purpose:
    Confirms that all core dependencies (OpenCV, MediaPipe, NumPy) are
    installed correctly and that the webcam can be accessed, BEFORE any
    pose-detection code is written.

    Run this once after setting up the virtual environment. If everything
    prints "OK", Phase 1 is complete and we can move to Phase 2.

Usage:
    python verify_setup.py
"""

import sys


def check_python_version() -> None:
    """Confirm the interpreter meets the Python 3.10+ requirement."""
    major, minor = sys.version_info[0], sys.version_info[1]
    print(f"Python version: {sys.version.split()[0]}", end=" -> ")
    if (major, minor) >= (3, 10):
        print("OK")
    else:
        print("FAIL (need Python 3.10 or higher)")


def check_library_versions() -> None:
    """Import each required library and print its version."""
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__} -> OK")
    except ImportError as e:
        print(f"OpenCV import FAILED: {e}")

    try:
        import mediapipe as mp
        print(f"MediaPipe version: {mp.__version__} -> OK")
    except ImportError as e:
        print(f"MediaPipe import FAILED: {e}")

    try:
        import numpy as np
        print(f"NumPy version: {np.__version__} -> OK")
    except ImportError as e:
        print(f"NumPy import FAILED: {e}")


def check_webcam_access() -> None:
    """
    Try to open the default webcam (index 0) and grab a single frame.
    Handles the case where no webcam is present or it's in use elsewhere,
    since we must fail gracefully rather than crash.
    """
    import cv2

    print("Checking webcam access...", end=" ")
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("FAIL (could not open webcam - check camera permissions/index)")
        return

    success, frame = capture.read()
    capture.release()

    if success and frame is not None:
        print(f"OK (frame size: {frame.shape[1]}x{frame.shape[0]})")
    else:
        print("FAIL (webcam opened but no frame was captured)")


if __name__ == "__main__":
    print("---- AI Fitness Trainer: Environment Verification ----\n")
    check_python_version()
    check_library_versions()
    check_webcam_access()
    print("\n---- Verification complete ----")