from types import SimpleNamespace

import numpy as np

from src.input_validation import (
    validate_human_subject,
    validate_image_content,
)


def _face(
    *,
    face_count=1,
    width=400,
    height=500,
    x0=100,
    x1=300,
    y0=80,
    y1=420,
):
    xs = np.linspace(x0, x1, 478)
    ys = np.linspace(y0, y1, 478)
    landmarks = list(zip(xs, ys))
    return SimpleNamespace(
        success=True,
        error=None,
        face_count=face_count,
        landmarks=landmarks,
    ), (height, width, 3)


def test_blank_image_is_rejected_before_face_detection():
    image = np.full((400, 300, 3), 127, dtype=np.uint8)

    result = validate_image_content(image)

    assert not result.valid
    assert result.code == "blank_or_uniform_image"
    assert "blank" in result.message.lower()


def test_nonuniform_rgb_image_passes_content_validation():
    gradient = np.tile(np.arange(0, 255, dtype=np.uint8), (300, 2))
    image = np.stack([gradient, np.flipud(gradient), gradient], axis=-1)

    result = validate_image_content(image)

    assert result.valid


def test_invalid_channel_shape_is_rejected():
    result = validate_image_content(np.zeros((200, 200), dtype=np.uint8))

    assert not result.valid
    assert result.code == "invalid_image_shape"


def test_no_detected_face_is_reported_as_nonhuman_content():
    face = SimpleNamespace(
        success=False,
        error="No face detected.",
        face_count=0,
        landmarks=None,
    )

    result = validate_human_subject(face, (500, 400, 3))

    assert not result.valid
    assert result.code == "no_human_face"
    assert "cars" in result.message.lower()


def test_multiple_people_are_rejected_as_ambiguous():
    face, shape = _face(face_count=2)

    result = validate_human_subject(face, shape)

    assert not result.valid
    assert result.code == "multiple_human_faces"
    assert "exactly one" in result.message.lower()


def test_single_large_in_frame_face_passes():
    face, shape = _face()

    result = validate_human_subject(face, shape)

    assert result.valid
    assert result.code == "single_human_face_valid"


def test_small_face_is_rejected_for_color_measurement():
    face, shape = _face(x0=180, x1=220, y0=220, y1=270)

    result = validate_human_subject(face, shape)

    assert not result.valid
    assert result.code == "face_too_small"
