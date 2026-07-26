# ShadeSense AI Demo Talking Points

## How The System Handles Real-World Variation

- Different skin tones: extraction uses adaptive per-region thresholds instead
  of fixed brightness cutoffs, so fair, medium, tan, deep, and rich-deep skin
  colors are evaluated relative to their own region.
- Deep-skin-safe adaptive filtering: dark pixels are not rejected just because
  their luminance is low. The filter looks for extremes within the same facial
  region.
- Shadows: shadow-like patches are rejected or reduced when they are extreme
  relative to the local region, while valid deeper skin tone remains usable.
- Facial highlights and specular shine: bright, washed-out, or high-contrast
  highlight patches are excluded so the final swatch is based more on diffuse
  skin tone than shine.
- Mild makeup or highlight influence: red/pink or unusually bright patch
  patterns are treated conservatively as possible influence. The app reduces
  weight and explains the uncertainty; it does not claim to remove makeup.
- Jawline and forehead contamination: cheeks are the primary anchor. Forehead
  can be excluded if it looks contaminated by hairline, shine, or shadow.
  Jawline can support shade depth when reliable, or be reduced when there are
  signs of chin/neck shadow, contour, occlusion, or uneven lighting.
- Color correction safeguard: original image color is preserved unless the
  corrected version improves extraction reliability without excessive Lab or
  chroma shift.
- Capture readiness vs candidate confidence: I separated photo/extraction
  reliability from evidence for each catalog shade. Poor lighting, weak face
  quality, few valid pixels, uneven region agreement, and cheek imbalance lower
  the global readiness ceiling instead of flattening every shade to one number.
- Candidate confidence: each shade varies below that ceiling using 65% color
  fit, 25% shade-family stability, and 10% catalog evidence. Missing evidence is
  omitted and the remaining weights are normalized.
- Public catalog limitations: catalog swatches are website-derived
  approximations. They may differ from applied foundation because of brand image
  processing, display calibration, oxidation, texture, and lighting.

## Why True Skin Color Is Hard From A Single Selfie

- Camera processing: phones and cameras apply sharpening, tone mapping,
  denoising, beautification, HDR, and compression that can alter color.
- Lighting: indoor bulbs, daylight, mixed lighting, and studio flashes all
  change the apparent skin tone.
- White balance: automatic white balance can push the image warmer, cooler, or
  greener than the real scene.
- Shadows and highlights: skin is not a flat color. Shine, oily areas, under-eye
  shadows, chin shadows, and directional light all create different observed
  colors on the same face.
- Makeup: foundation, concealer, blush, bronzer, highlighter, and lip color can
  contaminate the sampled regions.
- Website-derived catalog swatches: the catalog colors are approximations from
  public product/swatch data, not measurements of real product on skin.
- Candidate confidence is heuristic: the score is explainable and factor-based,
  but it is not a calibrated probability of real-world shade satisfaction.
- Skin Extraction Quality is also heuristic, but it answers a different
  question: whether the extracted skin swatch is reliable enough to compare
  against the catalog.
