"""Evaluate cross-photo skin-tone repeatability without storing personal data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture_uncertainty import analyze_capture_uncertainty
from src.color_correction import apply_mild_color_correction, correction_settings_for_lighting
from src.extraction_quality import build_extraction_quality_report
from src.extraction_selection import run_dual_extraction
from src.face_detection import detect_face_landmarks
from src.image_io import open_rgb_image
from src.image_quality import analyze_image_quality
from src.lighting_quality import analyze_lighting_quality
from src.lighting_sensitivity import analyze_lighting_sensitivity
from src.multicapture_consensus import CaptureEvidence, build_multicapture_consensus
from src.region_masks import build_region_masks, refine_masks_for_capture


def analyze_file(path: Path) -> tuple[CaptureEvidence | None, dict]:
    image = np.asarray(open_rgb_image(path))
    global_lighting = analyze_lighting_quality(image)
    provisional, _ = apply_mild_color_correction(
        image, **correction_settings_for_lighting(global_lighting)
    )
    face = detect_face_landmarks(provisional)
    if not face.success:
        return None, {"capture_id": path.name, "error": face.error}
    provisional_image_quality = analyze_image_quality(image, face.landmarks)
    masks = build_region_masks(image.shape, face.landmarks)
    masks, mask_diagnostics = refine_masks_for_capture(
        image, masks, face.landmarks, provisional_image_quality.pose_asymmetry
    )
    image_quality = analyze_image_quality(image, face.landmarks, masks=masks)
    lighting = analyze_lighting_quality(image, masks)
    corrected, _ = apply_mild_color_correction(
        image, **correction_settings_for_lighting(lighting)
    )
    selection = run_dual_extraction(image, corrected, masks, lighting, "auto")
    skin = selection.selected
    skin.capture_region_diagnostics = mask_diagnostics
    sensitivity_source = image if selection.selected_source == "original" else corrected
    sensitivity = analyze_lighting_sensitivity(sensitivity_source, masks, skin)
    skin.lighting_sensitivity_labs = sensitivity.variant_labs
    skin.lighting_sensitivity_diagnostics = sensitivity.as_diagnostics()
    uncertainty = analyze_capture_uncertainty(skin, lighting, image_quality)
    skin.systematic_uncertainty_diagnostics = uncertainty.as_diagnostics()
    extraction = build_extraction_quality_report(
        skin, image_quality, lighting, selection, face
    )
    lab = skin.foundation_target_lab if skin.foundation_target_active else skin.lab
    evidence = CaptureEvidence(
        capture_id=path.name,
        lab=tuple(float(value) for value in lab),
        extraction_score=float(extraction["overall_score"]),
        lighting_score=float(lighting.score),
        uncertainty_radius=uncertainty.total_radius,
        low_signal=lighting.low_signal,
    )
    return evidence, {
        "capture_id": path.name,
        "lab": evidence.lab,
        "extraction_score": evidence.extraction_score,
        "lighting_score": evidence.lighting_score,
        "uncertainty_radius": evidence.uncertainty_radius,
        "low_signal": evidence.low_signal,
        "recapture_recommended": lighting.recapture_recommended,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--pattern", default="Image_*.*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(path for path in args.folder.glob(args.pattern) if path.is_file())
    evidence = []
    captures = []
    for path in paths:
        item, report = analyze_file(path)
        captures.append(report)
        if item is not None:
            evidence.append(item)
    consensus = build_multicapture_consensus(evidence)
    payload = {
        "captures": captures,
        "consensus": {
            "success": consensus.success,
            "lab": consensus.lab,
            "anchor_capture_id": consensus.anchor_capture_id,
            "included_capture_ids": consensus.included_capture_ids,
            "excluded_capture_ids": consensus.excluded_capture_ids,
            "excluded_low_signal_capture_ids": (
                consensus.excluded_low_signal_capture_ids
            ),
            "excluded_perceptual_outlier_capture_ids": (
                consensus.excluded_perceptual_outlier_capture_ids
            ),
            "delta_e_by_capture": consensus.delta_e_by_capture,
            "uncertainty_radius_p90": consensus.uncertainty_radius_p90,
            "repeatability_score": consensus.repeatability_score,
            "warnings": consensus.warnings,
            "method": consensus.method,
        },
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if consensus.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
