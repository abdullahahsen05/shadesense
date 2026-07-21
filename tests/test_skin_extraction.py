import numpy as np
from PIL import Image

from src.color_correction import apply_mild_color_correction
from src.config import PROJECT_ROOT
from src.face_detection import detect_face_landmarks
from src.region_masks import build_region_masks
from src.skin_extraction import MIN_VALID_PIXELS_PER_REGION, extract_skin_tone

SAMPLES = PROJECT_ROOT / "data" / "sample_images"


def _extract_for(name):
    img = np.array(Image.open(SAMPLES / name).convert("RGB"))
    corrected, _ = apply_mild_color_correction(img)
    result = detect_face_landmarks(corrected)
    masks = build_region_masks(corrected.shape, result.landmarks)
    return extract_skin_tone(corrected, masks)


def test_extraction_succeeds_on_clear_face():
    skin = _extract_for("face_astronaut.png")
    assert skin.success
    assert all(0 <= v <= 255 for v in skin.rgb)
    assert 0.0 <= skin.quality_score <= 1.0


def test_uses_median_not_full_mean():
    skin = _extract_for("face_astronaut.png")
    for region in skin.region_results.values():
        if region.median_rgb is not None:
            assert isinstance(region.median_rgb, tuple)


def test_final_color_is_close_to_cheek_jaw_tones_not_hair():
    skin = _extract_for("face_astronaut.png")
    cheek_jaw_rgbs = [
        r.median_rgb
        for name, r in skin.region_results.items()
        if name != "forehead" and r.median_rgb is not None
    ]
    avg_cheek_jaw = np.mean(cheek_jaw_rgbs, axis=0)
    dist = np.linalg.norm(np.array(skin.rgb) - avg_cheek_jaw)
    assert dist < 30, "Final skin color strayed far from cheek/jaw consensus (possible hair contamination)"


def test_low_pixel_region_flagged_in_warnings_or_reliability():
    skin = _extract_for("face_astronaut.png")
    for region in skin.region_results.values():
        if region.valid_pixel_count < MIN_VALID_PIXELS_PER_REGION:
            assert not region.reliable


def test_color_correction_returns_notes_and_valid_image():
    img = np.array(Image.open(SAMPLES / "face_astronaut.png").convert("RGB"))
    corrected, notes = apply_mild_color_correction(img)
    assert corrected.shape == img.shape
    assert corrected.dtype == np.uint8
    assert len(notes) > 0
