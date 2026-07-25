import cv2
import numpy as np

from src.image_quality import analyze_image_quality


def _sharp_checkerboard(size=160):
    grid = np.indices((size, size)).sum(axis=0) % 2
    image = (grid * 255).astype(np.uint8)
    return np.dstack([image, image, image])


def test_blur_score_lower_for_blurred_than_sharp_image():
    sharp = _sharp_checkerboard()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

    sharp_quality = analyze_image_quality(sharp)
    blurred_quality = analyze_image_quality(blurred)

    assert blurred_quality.blur_score < sharp_quality.blur_score


def test_overexposure_warning_for_bright_image():
    image = np.full((120, 120, 3), 250, dtype=np.uint8)
    quality = analyze_image_quality(image)

    assert "Strong highlights or overexposure detected." in quality.warnings
    assert 0 <= quality.overall_score <= 100


def test_underexposure_warning_for_dark_image():
    image = np.full((120, 120, 3), 12, dtype=np.uint8)
    quality = analyze_image_quality(image)

    assert "Image appears underexposed." in quality.warnings
    assert 0 <= quality.overall_score <= 100


def test_face_mask_prevents_dark_background_from_triggering_underexposure():
    image = np.full((160, 160, 3), 12, dtype=np.uint8)
    image[45:120, 50:110] = (170, 145, 125)
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[45:120, 50:110] = 255

    quality = analyze_image_quality(image, masks={"combined": mask})

    assert "Image appears underexposed." not in quality.warnings
    assert quality.exposure_source == "facial skin regions"
    assert quality.underexposed_ratio == 0.0


def test_face_mask_detects_dark_face_despite_bright_background():
    image = np.full((160, 160, 3), 180, dtype=np.uint8)
    image[45:120, 50:110] = (15, 15, 15)
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[45:120, 50:110] = 255

    quality = analyze_image_quality(image, masks={"combined": mask})

    assert "Image appears underexposed." in quality.warnings
    assert quality.underexposed_ratio == 1.0


def test_strong_color_cast_warning_for_tinted_image():
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = (180, 80, 65)

    quality = analyze_image_quality(image)

    assert "Possible color cast detected." in quality.warnings
    assert quality.color_cast_score < 65


def test_overall_score_stays_in_range_with_landmarks():
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    landmarks = [(60, 60), (140, 60), (100, 100), (70, 145), (130, 145)]

    quality = analyze_image_quality(image, landmarks)

    assert 0 <= quality.overall_score <= 100
    assert quality.label in {"excellent", "good", "fair", "poor"}
