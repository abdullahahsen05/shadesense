# Approach

ShadeSense AI is an explainable, non-ML-classifier pipeline for foundation shade
recommendation:

```text
input image
-> provisional mild correction for landmark stability
-> face detection / landmarks (MediaPipe Face Landmarker, 478 points)
-> forehead / left cheek / right cheek / jawline region masks
   (derived geometrically from trusted landmark groups: face oval, eyes,
   eyebrows, lips, nose; not hardcoded per-region index lists)
-> face-region lighting analysis and final conservative correction
-> per-region pixel filtering (luminance percentile trim, saturation trim)
-> adaptive stable patches with local highlight/shadow metrics
-> weighted CIEDE2000 medoid + MAD outlier rejection with region caps
-> 96-sample stratified patch bootstrap for Lab uncertainty
-> deterministic exposure / white-balance / gamma sensitivity re-extraction
-> evidence-gated foundation target L* with cheek-derived undertone
-> Skin Extraction Quality score (image capture, region reliability, patch
   stability, lighting safety, color consistency, and region stability)
-> distribution-aware CIEDE2000 ranking using central, median, and tail distances,
   with close-tie-only depth, supported-lightness, and catalog-evidence adjustments
-> Top 3 recommendations evaluated across bootstrap and lighting-sensitivity samples
-> global capture readiness + candidate-specific confidence
-> deterministic explanation text
```

No labeled training data or ML classifier is used for shade prediction. Every
step is deterministic and inspectable, which matters both because no labeled
shade dataset exists and because it lets each recommendation be explained in
plain language.

Skin Extraction Quality, Capture Readiness, and Candidate Confidence are
intentionally separate:

- Skin Extraction Quality asks, "how reliable was the color extraction?"
- Capture Readiness asks, "is this photo safe to use for recommendations?"
- Candidate Confidence asks, "how strong is the evidence for this particular
  shade within the capture's safety ceiling?"

`ready`, `caution`, and `provisional` all return Top 3. Their confidence ceilings
are 93%, 75%, and 55%, so a poor image cannot produce falsely strong candidate
scores. Candidates still vary below that ceiling according to their own color
fit, shade-family stability, and catalog evidence.

See `docs/limitations.md` for known gaps in the current local build.
