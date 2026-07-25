from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.multi_photo_consensus import build_multi_photo_consensus
from src.recommendation_readiness import RecommendationReadiness
from src.skin_extraction import SkinToneResult


def _analysis(lab, score=80, state="ready"):
    skin = SkinToneResult(
        rgb=(150, 110, 90),
        lab=tuple(lab),
        region_results={},
        quality_score=score / 100,
        region_consistency=0.9,
        avg_valid_pixel_ratio=0.8,
        usable_region_count=3,
        bootstrap_labs=[tuple(lab)] * 8,
        uncertainty_diagnostics={
            "stability_score": 90,
            "delta_e_radius_p90": 2,
        },
    )
    readiness = RecommendationReadiness(
        state=state,
        score=score,
        confidence_cap=0.9,
        summary="test",
        capture_readiness_score=score,
    )
    return SimpleNamespace(
        success=True,
        skin_result=skin,
        extraction_quality_report={"overall_score": score},
        recommendation_readiness=readiness,
        lighting_quality=SimpleNamespace(
            score=0.85,
            low_signal=False,
            warnings=[],
        ),
        face_result=SimpleNamespace(warnings=[]),
    )


def _catalog():
    return pd.DataFrame(
        [
            {
                "shade_id": "a",
                "brand": "Brand",
                "shade_name": "A",
                "hex": "#96705A",
                "r": 150,
                "g": 112,
                "b": 90,
                "lab_l": 50.0,
                "lab_a": 10.0,
                "lab_b": 16.0,
                "catalog_quality_score": 0.8,
                "product_type": "foundation",
                "depth": "medium",
            },
            {
                "shade_id": "b",
                "brand": "Brand",
                "shade_name": "B",
                "hex": "#A47A62",
                "r": 164,
                "g": 122,
                "b": 98,
                "lab_l": 56.0,
                "lab_a": 10.0,
                "lab_b": 16.0,
                "catalog_quality_score": 0.8,
                "product_type": "foundation",
                "depth": "medium",
            },
            {
                "shade_id": "c",
                "brand": "Other",
                "shade_name": "C",
                "hex": "#855C48",
                "r": 133,
                "g": 92,
                "b": 72,
                "lab_l": 44.0,
                "lab_a": 10.0,
                "lab_b": 16.0,
                "catalog_quality_score": 0.8,
                "product_type": "foundation",
                "depth": "tan",
            },
        ]
    )


def test_consensus_rejects_one_gross_outlier_and_is_deterministic():
    analyses = [
        _analysis((50.0, 10.0, 16.0)),
        _analysis((50.8, 10.2, 15.7)),
        _analysis((75.0, -5.0, 3.0), score=55, state="provisional"),
    ]

    first = build_multi_photo_consensus(analyses, _catalog())
    second = build_multi_photo_consensus(analyses, _catalog())

    assert first.success
    assert first.retained_indices == [0, 1]
    assert first.rejected_indices == [2]
    assert first.matches[0].shade_id == "a"
    assert first.consensus_lab == second.consensus_lab
    assert first.uncertainty_labs == second.uncertainty_labs


def test_two_disagreeing_photos_are_not_silently_dropped():
    result = build_multi_photo_consensus(
        [
            _analysis((45.0, 8.0, 14.0)),
            _analysis((62.0, 8.0, 14.0)),
        ],
        _catalog(),
    )

    assert result.retained_indices == [0, 1]
    assert result.rejected_indices == []
    assert result.readiness.state == "provisional"
    assert result.readiness.confidence_cap == 0.55
