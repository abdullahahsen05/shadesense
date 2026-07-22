# Approach

ShadeSense AI is an explainable, non-ML-classifier pipeline for foundation shade
recommendation:

```text
input image
-> mild lighting correction (gray-world white balance + luminance CLAHE)
-> face detection / landmarks (MediaPipe Face Landmarker, 478 points)
-> forehead / left cheek / right cheek / jawline region masks
   (derived geometrically from trusted landmark groups: face oval, eyes,
   eyebrows, lips, nose; not hardcoded per-region index lists)
-> per-region pixel filtering (luminance percentile trim, saturation trim)
-> per-region median RGB/Lab + outlier-region exclusion
-> robust patch-level voting + region stability checks into one representative
   skin Lab color
-> Skin Extraction Quality score (image capture, region reliability, patch
   stability, lighting safety, color consistency, and region stability)
-> CIEDE2000 distance against the shade catalog (Lab space)
-> Top 3 recommendations
-> Match confidence (catalog color distance, region consistency, valid-pixel
   ratio, lighting, face quality, top1/top2 separation) + deterministic
   explanation text
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

See `docs/limitations.md` for known gaps in the current local build.
