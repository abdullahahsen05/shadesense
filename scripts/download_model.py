"""Download the MediaPipe Face Landmarker model into models/face_landmarker.task.

Run once after installing dependencies:
    python scripts/download_model.py
"""

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Model already present at {DEST}")
        return
    print(f"Downloading {MODEL_URL} -> {DEST}")
    urllib.request.urlretrieve(MODEL_URL, DEST)
    print("Done.")


if __name__ == "__main__":
    main()
