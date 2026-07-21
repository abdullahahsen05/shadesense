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
  region toward hair color. An outlier-region safeguard in
  `src/skin_extraction.py` detects and excludes a region whose color
  disagrees strongly with the others (requires at least 3 usable regions),
  which mitigates but does not eliminate this failure mode — with only 2
  usable regions it cannot reliably distinguish "hair" from "skin."
- No explicit handling for glasses, heavy occlusion, or facial hair beyond
  what the region geometry and outlier-region logic naturally avoid.

## Skin tone extraction
- Uses percentile-based luminance/saturation filtering rather than a full
  skin-color probability model; extreme makeup (heavy contour/blush) can
  still shift the extracted tone.
- Mild color correction (gray-world + light CLAHE) is intentionally
  conservative; it does not correct strong color casts (e.g. colored indoor
  lighting) beyond a moderate amount.

## Shade catalog and matching
- Ships with a mock catalog (`data/shade_catalog_mock.csv`, 18 shades). Real
  catalogs with more shades and richer metadata will likely improve match
  quality once available.
- CIEDE2000 in `skimage` does not correctly broadcast a single Lab color
  against an array of catalog Lab colors — `src/shade_matcher.py` explicitly
  tiles the skin color before calling it (covered by a regression test in
  `tests/test_shade_matcher.py`).

## Confidence
- Confidence is a heuristic, interpretable weighted combination (match
  distance 50%, region consistency 20%, valid pixel ratio 10%, face quality
  10%, top1/top2 separation 10%), not a calibrated statistical probability.
  It is capped below 93% by design so the app never claims near-certainty.

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
