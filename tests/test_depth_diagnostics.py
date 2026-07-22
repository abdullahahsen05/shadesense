import numpy as np
import pandas as pd

from src.depth_diagnostics import (
    calculate_ita,
    depth_match_status,
    depth_sanity_note,
    estimate_depth_from_ita,
)
from src.shade_matcher import match_shades
from src.skin_extraction import extract_skin_tone


def _simple_masks():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    masks = {
        "forehead": np.zeros((80, 80), dtype=np.uint8),
        "left_cheek": np.zeros((80, 80), dtype=np.uint8),
        "right_cheek": np.zeros((80, 80), dtype=np.uint8),
        "jawline": np.zeros((80, 80), dtype=np.uint8),
    }
    image[0:18, :] = (160, 110, 80)
    masks["forehead"][0:18, :] = 255
    image[24:44, 0:40] = (160, 110, 80)
    masks["left_cheek"][24:44, 0:40] = 255
    image[24:44, 40:80] = (160, 110, 80)
    masks["right_cheek"][24:44, 40:80] = 255
    image[52:72, :] = (145, 96, 70)
    masks["jawline"][52:72, :] = 255
    return image, masks


def test_ita_calculation_and_depth_category():
    ita = calculate_ita(50.0, 20.0)
    assert abs(ita) < 1e-6

    ita_label, depth = estimate_depth_from_ita(ita)
    assert ita_label == "brown"
    assert depth == "deep"


def test_depth_match_status_and_note():
    assert depth_match_status("deep", "deep") == "aligned"
    assert depth_match_status("deep", "tan") == "close"
    assert depth_match_status("deep", "light") == "possible mismatch"
    assert "color distance was still close" in depth_sanity_note("deep", "light")


def test_ita_depth_diagnostic_appears_in_extraction_result():
    image, masks = _simple_masks()
    skin = extract_skin_tone(image, masks)

    assert skin.ita_degrees is not None
    assert skin.ita_category
    assert skin.depth_estimate in {"medium", "tan", "deep"}


def test_depth_sanity_note_does_not_override_better_delta_e_match():
    df = pd.DataFrame(
        {
            "shade_id": ["best_mismatch", "worse_aligned"],
            "brand": ["T", "T"],
            "shade_name": ["Best Color", "Worse Depth"],
            "hex": ["#8C6446", "#A87858"],
            "r": [140, 168],
            "g": [100, 120],
            "b": [70, 88],
            "lab_l": [45.0, 55.0],
            "lab_a": [8.0, 9.0],
            "lab_b": [18.0, 18.0],
            "depth": ["light", "tan"],
        }
    )

    matches = match_shades(np.array([45.0, 8.0, 18.0]), df, top_k=2)

    assert matches[0].shade_name == "Best Color"
    assert matches[0].depth_match_status == "possible mismatch"
    assert "color distance was still close" in matches[0].depth_sanity_note
