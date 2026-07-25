# Known Limitations (Local Build)

## Face detection
- Uses MediaPipe's short-range Face Landmarker model, optimized for
  close-up/selfie-style photos. Faces that occupy a small fraction of the
  frame (e.g. group photos, distant subjects) may not be detected at all —
  this is a property of the underlying model, not a bug in this app.
- Very wide/non-square aspect ratios (roughly beyond ~1.8:1) reduce
  detection reliability because the model letterboxes to a square input
  internally, shrinking the effective face size. Typical phone-camera
  aspect ratios (up to ~16:9) work fine in testing.
- Multiple-face handling picks the largest/most central face; it does not
  attempt to let the user choose which face to analyze.

## Region masks
- Forehead/cheek/jawline masks are derived geometrically from landmark
  groups (face oval, eyes, eyebrows, lips, nose) rather than hand-tuned
  per-region polygons, so they generalize reasonably across face shapes and
  poses but are not pixel-perfect for every hairstyle or face angle.
- Heavy bangs/fringes covering the forehead can still bias the forehead
  region toward hair color. `src/skin_extraction.py` treats the cheeks as
  the trust anchor (least likely to be occluded by hair or facial-hair
  shadow): forehead is excluded outright when it disagrees strongly with
  the cheek tone (likely hair/fringe or shadow), and jawline is
  down-weighted (not excluded) when it is specifically darker than the
  cheeks (possible facial hair or chin shadow). This requires at least one
  reliable cheek region to act as the anchor — with neither cheek usable,
  these checks are skipped entirely (no anchor to compare against).
- No explicit handling for glasses or heavy occlusion beyond what the
  region geometry and the forehead/jawline trust-anchor logic naturally
  mitigate.

## Skin tone extraction
- Uses percentile-based luminance/saturation filtering rather than a full
  skin-color probability model; extreme makeup (heavy contour/blush) can
  still shift the extracted tone.
- Mild color correction (gray-world + light CLAHE) is intentionally
  conservative; it does not correct strong color casts (e.g. colored indoor
  lighting) beyond a moderate amount.
- Bootstrap intervals quantify disagreement between retained patches, but they
  are internal resampling uncertainty rather than calibrated real-world error
  bounds. Dataset validation with measured skin color is still required.

## Shade catalog and matching
- Ships with a mock catalog (`data/shade_catalog_mock.csv`, 18 shades). Real
  catalogs with more shades and richer metadata will likely improve match
  quality once available.
- CIEDE2000 in `skimage` does not correctly broadcast a single Lab color
  against an array of catalog Lab colors — `src/shade_matcher.py` explicitly
  tiles the skin color before calling it (covered by a regression test in
  `tests/test_shade_matcher.py`).

## Confidence
- Confidence is a heuristic, interpretable combination of match distance,
  region/pixel reliability, face-aware lighting, face quality, patch-bootstrap
  uncertainty, recommendation stability, catalog evidence, and Top-1/Top-2
  separation. It is not a calibrated statistical probability.
- Confidence is capped below 93% by design. `caution` recommendations are
  capped at 75% and `provisional` recommendations at 55%, while still returning
  the required Top 3 candidates.

## Evaluation evidence

- MST-E and FairFace do not provide measured physical foundation swatches or
  verified wearer-to-product labels. They measure face detection, extraction
  repeatability, robustness, recapture behavior, and recommendation stability,
  not exact applied-product accuracy.
- Multi-photo consensus can reduce capture variation but cannot correct three
  consistently biased photos. Two strongly disagreeing photos remain
  provisional because neither can safely be identified as the outlier.
- Neutral-card calibration requires a genuinely neutral card photographed with
  the same camera settings and illumination. A screen, white paper, or a card
  in different lighting can introduce rather than remove color error.
- ICC conversion honors embedded profiles, but images without a profile must be
  treated as sRGB by convention.

## Testing
- Automated tests use a single real photographic face (scikit-image's
  bundled `astronaut` sample, bundled offline — no network image downloads
  were made) plus flipped/rotated/brightness-adjusted variants of it to
  exercise mask geometry and lighting robustness. This was a deliberate
  choice to avoid downloading additional face photos without asking; a
  broader, diverse-identity test set (skin tones, ages, hairstyles) was not
  available offline and would strengthen confidence in generalization.
- No dedicated performance/latency testing was done; MediaPipe model
  loading happens once per Streamlit session (cached), not per image.

## Out of scope for this pass (by design)
Deployment, cloud hosting, FastAPI/backend service, database, auth, demo
polish, demo QA, and submission packaging are explicitly out of scope for
this local-app build pass (see `CLAUDE.md`).
