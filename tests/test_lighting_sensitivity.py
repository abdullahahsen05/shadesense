from types import SimpleNamespace

import numpy as np

import src.lighting_sensitivity as sensitivity_module
from src.lighting_sensitivity import (
    analyze_lighting_sensitivity,
    generate_lighting_perturbations,
)


def _baseline(lab=(50.0, 8.0, 14.0)):
    return SimpleNamespace(
        success=True,
        lab=lab,
        foundation_target_active=False,
        foundation_target_lab=None,
    )


def test_lighting_perturbations_are_deterministic_and_conservative():
    image = np.full((40, 40, 3), [150, 110, 85], dtype=np.uint8)

    first = generate_lighting_perturbations(image)
    second = generate_lighting_perturbations(image)

    assert list(first) == [
        "exposure_darker",
        "exposure_brighter",
        "white_balance_warm",
        "white_balance_cool",
        "gamma_shadow_deeper",
        "gamma_shadow_lighter",
    ]
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert all(value.shape == image.shape for value in first.values())
    assert all(value.dtype == np.uint8 for value in first.values())
    assert max(
        float(np.mean(np.abs(value.astype(float) - image.astype(float))))
        for value in first.values()
    ) < 12.0


def test_stable_variant_extractions_produce_high_sensitivity_score(monkeypatch):
    image = np.full((40, 40, 3), [150, 110, 85], dtype=np.uint8)
    masks = {"combined": np.full((40, 40), 255, dtype=np.uint8)}

    monkeypatch.setattr(
        sensitivity_module,
        "extract_skin_tone",
        lambda _image, _masks: _baseline(),
    )

    result = analyze_lighting_sensitivity(image, masks, _baseline())

    assert result.successful_variants == result.attempted_variants == 6
    assert result.delta_e_p90 == 0.0
    assert result.score == 100.0
    assert result.stable
    assert result.warnings == []


def test_sensitive_variant_extractions_warn_and_reduce_score(monkeypatch):
    image = np.full((40, 40, 3), [150, 110, 85], dtype=np.uint8)
    masks = {"combined": np.full((40, 40), 255, dtype=np.uint8)}

    def sensitive_extractor(variant, _masks):
        channel_gap = float(variant[:, :, 0].mean() - variant[:, :, 2].mean())
        return _baseline((50.0 + 0.8 * channel_gap, 8.0, 14.0))

    monkeypatch.setattr(
        sensitivity_module,
        "extract_skin_tone",
        sensitive_extractor,
    )

    result = analyze_lighting_sensitivity(image, masks, _baseline())

    assert result.delta_e_p90 > 6.0
    assert result.score < 40.0
    assert not result.stable
    assert any("recapture" in warning.lower() for warning in result.warnings)
