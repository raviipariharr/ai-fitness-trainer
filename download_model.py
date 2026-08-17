"""
download_model.py

Purpose:
    Downloads the MediaPipe PoseLandmarker model file (~5-9 MB) into
    models/pose_landmarker_lite.task. Run this once before using
    pose detection. The model is not committed to the repo since it's
    a binary asset you can always re-download.

Usage:
    python download_model.py
"""

import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")


def download_model() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}. Skipping download.")
        return

    print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        print(f"Download failed: {error}")
        print(f"You can also download it manually from:\n{MODEL_URL}")
        return

    print("Download complete.")


if __name__ == "__main__":
    download_model()