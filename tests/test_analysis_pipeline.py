from pathlib import Path

import numpy as np

from src.analysis_pipeline import (
    analyze_rgb_image,
    normalize_analysis_resolution,
)
from src.image_io import open_rgb_image
from src.shade_catalog import MOCK_CATALOG_KEY, load_named_catalog


SAMPLES = Path("data/sample_images")


def test_shared_pipeline_returns_complete_face_analysis():
    image_rgb = np.asarray(open_rgb_image(SAMPLES / "face_astronaut.png"))
    catalog = load_named_catalog(MOCK_CATALOG_KEY)

    result = analyze_rgb_image(image_rgb, catalog)

    assert result.success
    assert result.face_result.success
    assert {
        "forehead",
        "left_cheek",
        "right_cheek",
        "jawline",
    }.issubset(result.masks)
    assert result.skin_result.success
    assert result.extraction_quality_report["overall_score"] >= 0
    assert result.recommendation_readiness.state in {
        "ready",
        "caution",
        "provisional",
    }
    assert len(result.matches) == 3
    assert all(match.confidence is not None for match in result.matches)


def test_shared_pipeline_returns_graceful_no_face_result():
    image_rgb = np.asarray(open_rgb_image(SAMPLES / "no_face_cat.png"))

    result = analyze_rgb_image(image_rgb)

    assert not result.success
    assert not result.face_result.success
    assert result.error
    assert result.masks == {}
    assert result.matches == []


def test_analysis_resolution_is_bounded_without_changing_aspect_ratio():
    image = np.zeros((2000, 1000, 3), dtype=np.uint8)

    resized, scale = normalize_analysis_resolution(image, max_side=1600)

    assert resized.shape == (1600, 800, 3)
    assert scale == 0.8
