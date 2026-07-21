# Approach

ShadeSense AI is an explainable, non-ML-classifier pipeline for foundation shade
recommendation:

```text
input image
→ mild lighting correction (gray-world white balance + luminance CLAHE)
→ face detection / landmarks (MediaPipe Face Landmarker, 478 points)
→ forehead / left cheek / right cheek / jawline region masks
  (derived geometrically from trusted landmark groups: face oval, eyes,
  eyebrows, lips, nose — not hardcoded per-region index lists)
→ per-region pixel filtering (luminance percentile trim, saturation trim)
→ per-region median RGB/Lab + outlier-region exclusion (guards against a
  region landing on hair/shadow rather than skin)
→ weighted combination into one representative skin Lab color
→ CIEDE2000 distance against the shade catalog (Lab space)
→ Top 3 recommendations
→ confidence (match distance, region consistency, valid-pixel ratio, face
  quality, top1/top2 separation) + deterministic explanation text
```

No labeled training data or ML classifier is used for shade prediction — every
step is deterministic and inspectable, which matters both because no labeled
shade dataset exists and because it lets each recommendation be explained in
plain language.

See `docs/limitations.md` for known gaps in the current local build.
