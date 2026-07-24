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
-> evidence-gated foundation target L* with cheek-derived undertone
-> Skin Extraction Quality score (image capture, region reliability, patch
   stability, lighting safety, color consistency, and region stability)
-> CIEDE2000 distance against the shade catalog (Lab space) with close-tie-only
   depth, supported-lightness, and catalog-evidence adjustments
-> Top 3 recommendations retained across bootstrap samples
-> readiness-aware Match confidence + deterministic explanation text
```

No labeled training data or ML classifier is used for shade prediction. Every
step is deterministic and inspectable, which matters both because no labeled
shade dataset exists and because it lets each recommendation be explained in
plain language.

Skin Extraction Quality and Match confidence are intentionally separate. Skin
Extraction Quality asks, "is this extracted swatch reliable?" Match confidence
asks, "does this reliable-or-unreliable swatch clearly match a catalog shade?"
Keeping them separate prevents the app from sounding overly certain when the
input image is poor.

The readiness state is also separate. It does not hide recommendations:
`ready`, `caution`, and `provisional` all return Top 3, but caution/provisional
states cap confidence and explicitly request a better capture.

See `docs/limitations.md` for known gaps in the current local build.
