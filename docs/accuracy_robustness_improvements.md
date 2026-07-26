# Accuracy and Robustness Improvements

This pass strengthens the local, explainable ShadeSense AI pipeline without
adding a shade classifier or another face model.

## What changed

- Lighting is evaluated on facial skin regions after landmarks are available.
  Bright or dark backgrounds no longer determine the primary lighting score.
- Facial masks use face-relative safety margins, and a foreshortened cheek is
  reduced based on visible area rather than skin brightness.
- Patch size scales with region resolution. Stable patches carry provenance,
  local variation, contrast, highlight, and shadow evidence.
- Patch consensus uses a weighted CIEDE2000 medoid and robust MAD thresholds.
  Forehead, jawline, and individual cheeks have explicit influence caps.
- Ninety-six deterministic stratified bootstrap samples estimate Lab
  uncertainty and recommendation stability.
- Foundation target L* changes only when highlights and agreeing lower-face
  regions support the shift. The adjustment is capped by depth and preserves
  cheek-derived undertone.
- Raw CIEDE2000 remains the primary catalog ranking. Depth, supported
  lightness, and catalog-quality terms can only resolve close candidates.
- Catalog HEX values are never heuristically calibrated. Product type and
  metadata completeness describe evidence quality, not physical product color.
- Readiness states always retain Top 3 recommendations while setting a global
  93%/75%/55% confidence ceiling. Candidate-specific scores vary below that
  ceiling using color fit, shade-family stability, and catalog evidence.
- Automatic correction is rejected when highlighted facial skin would be
  brightened by more than 3 L*, total skin lightness would shift by more than
  8 L*, or the extracted a*/b* undertone would shift by more than 7 units.
- Jaw evidence is limited to lateral side-jaw bands. A clean darker side jaw
  can support depth when it corroborates a cheek; off-undertone or contaminated
  jaw evidence is retained only as a diagnostic with minimal influence.
- Leave-one-region-out diagnostics distinguish limited independent support
  from genuine disagreement between trusted cheeks.
- Near-equivalent catalog colors are grouped into perceptual shade families
  after exact-product stability is measured. This improves Top-3 diversity
  without inflating certainty in a particular SKU.

## Readiness thresholds

| State | Minimum evidence | Confidence cap |
|---|---|---:|
| Ready | Extraction ≥70, lighting ≥65%, uncertainty radius ≤6 Delta E | 93% |
| Caution | Extraction ≥50, uncertainty radius ≤10 Delta E | 75% |
| Provisional | Anything weaker | 55% |

These scores remain engineering heuristics. A diverse dataset with measured
skin Lab values and verified foundation matches is required before reporting
real-world accuracy or statistically calibrated confidence.
