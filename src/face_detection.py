"""Face detection and landmark extraction using MediaPipe Face Landmarker."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image as MPImage
from mediapipe import ImageFormat

from src.config import PROJECT_ROOT, MIN_FACE_SIZE_RATIO

MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

NUM_LANDMARKS = 478


@dataclass
class FaceDetectionResult:
    success: bool
    landmarks: list | None  # list of (x, y) pixel-space tuples for the selected face
    all_faces_landmarks: list = field(default_factory=list)  # landmarks for every detected face
    face_count: int = 0
    image_shape: tuple | None = None
    warnings: list = field(default_factory=list)
    error: str | None = None


_landmarker = None


def _get_landmarker(num_faces: int = 5):
    """Lazily create and cache a FaceLandmarker instance."""
    global _landmarker
    if _landmarker is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {MODEL_PATH}. "
                "Expected the MediaPipe face_landmarker.task file to be present."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _select_face(faces_px: list, image_shape: tuple) -> int:
    """Select the largest/most central face when multiple are detected."""
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def score(landmarks):
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        fx, fy = sum(xs) / len(xs), sum(ys) / len(ys)
        centrality_penalty = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
        return area - centrality_penalty

    scores = [score(landmarks) for landmarks in faces_px]
    return int(np.argmax(scores))


def detect_face_landmarks(image_rgb: np.ndarray) -> FaceDetectionResult:
    """Detect face(s) and return pixel-space landmarks for the primary face.

    Prefers a single, largest/most-central face. Handles no-face and
    multiple-face cases gracefully without raising.
    """
    warnings: list = []
    h, w = image_rgb.shape[:2]

    if h < 60 or w < 60:
        return FaceDetectionResult(
            success=False,
            landmarks=None,
            face_count=0,
            image_shape=image_rgb.shape,
            warnings=warnings,
            error="Image is too small to reliably detect a face.",
        )

    try:
        landmarker = _get_landmarker()
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = landmarker.detect(mp_image)
    except Exception as exc:  # pragma: no cover - defensive against runtime/model errors
        return FaceDetectionResult(
            success=False,
            landmarks=None,
            face_count=0,
            image_shape=image_rgb.shape,
            warnings=warnings,
            error=f"Face detection failed: {exc}",
        )

    face_landmarks_list = result.face_landmarks
    face_count = len(face_landmarks_list)

    if face_count == 0:
        return FaceDetectionResult(
            success=False,
            landmarks=None,
            face_count=0,
            image_shape=image_rgb.shape,
            warnings=warnings,
            error="No face detected. Please upload a clear, front-facing photo.",
        )

    all_faces_px = []
    for face_landmarks in face_landmarks_list:
        px = [(lm.x * w, lm.y * h) for lm in face_landmarks]
        all_faces_px.append(px)

    if face_count > 1:
        warnings.append(
            f"{face_count} faces detected. Using the largest, most central face for analysis."
        )

    selected_idx = _select_face(all_faces_px, image_rgb.shape) if face_count > 1 else 0
    selected_landmarks = all_faces_px[selected_idx]

    xs = [p[0] for p in selected_landmarks]
    ys = [p[1] for p in selected_landmarks]
    face_w = max(xs) - min(xs)
    face_h = max(ys) - min(ys)
    if face_w < w * MIN_FACE_SIZE_RATIO or face_h < h * MIN_FACE_SIZE_RATIO:
        warnings.append(
            "Detected face is small relative to the image; results may be less reliable."
        )

    return FaceDetectionResult(
        success=True,
        landmarks=selected_landmarks,
        all_faces_landmarks=all_faces_px,
        face_count=face_count,
        image_shape=image_rgb.shape,
        warnings=warnings,
        error=None,
    )
