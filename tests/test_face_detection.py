import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT
from src.face_detection import detect_face_landmarks

SAMPLES = PROJECT_ROOT / "data" / "sample_images"


def _load(name):
    return np.array(Image.open(SAMPLES / name).convert("RGB"))


def test_detects_single_face():
    result = detect_face_landmarks(_load("face_astronaut.png"))
    assert result.success
    assert result.face_count == 1
    assert len(result.landmarks) == 478


def test_no_face_image_fails_gracefully():
    result = detect_face_landmarks(_load("no_face_cat.png"))
    assert not result.success
    assert result.error is not None


def test_multi_face_image_warns_and_selects_one():
    result = detect_face_landmarks(_load("multi_face.png"))
    assert result.success
    assert result.face_count >= 2
    assert any("faces detected" in w for w in result.warnings)
    assert len(result.landmarks) == 478
