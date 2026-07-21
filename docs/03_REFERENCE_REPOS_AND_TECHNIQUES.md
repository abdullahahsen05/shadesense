# 03 — Reference Repos and Techniques

These references are for inspiration and implementation guidance. Do not copy blindly. Use the ideas that support the assessment requirements.

## Primary References

### 1. MediaPipe Face Landmarker / Face Mesh

Links:
- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md

Use for:
- Face detection.
- Face landmarks.
- Creating cheek, forehead, and jawline masks.

Why useful:
- MediaPipe provides many face landmarks from a single image.
- It is robust and fast enough for a local Streamlit demo.
- It avoids training a custom model.

Implementation note:
- Start with MediaPipe Face Mesh or Face Landmarker.
- Convert normalized landmarks to pixel coordinates.
- Build polygon masks using selected landmark indices.

### 2. scikit-image color tools

Links:
- https://scikit-image.org/docs/stable/api/skimage.color.html

Use for:
- RGB to Lab conversion.
- CIEDE2000 color difference using `deltaE_ciede2000`.

Why useful:
- Foundation shade matching is a perceptual color-matching problem.
- Lab/Delta E is more defensible than raw RGB distance.

### 3. zschuessler/DeltaE

Link:
- https://github.com/zschuessler/DeltaE

Use for:
- Understanding Delta E formulas.
- Later JavaScript/Vercel conversion if the project moves to Next.js.

Do not use in Phase 1 unless we move to JS.

### 4. lovro-i/CIEDE2000

Link:
- https://github.com/lovro-i/CIEDE2000

Use for:
- Cross-checking CIEDE2000 formula behavior in Python.
- Educational reference if scikit-image output needs verification.

Prefer scikit-image in production code.

### 5. digantgarude/Skintone-Detector

Link:
- https://github.com/digantgarude/Skintone-Detector

Use for:
- Understanding simple skin-tone extraction approaches.

Important warning:
- This type of project is likely too simple for the final submission if it only detects general skin tone ranges.
- Our implementation must be stronger: landmarks, region masks, robust filtering, Lab/Delta E, confidence scoring, Top 3 recommendations.

### 6. yakhyo/face-parsing

Link:
- https://github.com/yakhyo/face-parsing

Use for optional advanced phase:
- Semantic face segmentation.
- Excluding hair, lips, eyes, eyebrows, and background more accurately.

Why useful:
- Face parsing can produce more precise skin masks than landmarks alone.

Warning:
- This may add PyTorch/ONNX dependency complexity.
- Keep this optional until the base MediaPipe pipeline works.

### 7. zllrunning/face-parsing.PyTorch

Link:
- https://github.com/zllrunning/face-parsing.PyTorch

Use for optional advanced phase:
- Reference for BiSeNet face parsing.
- Potential skin mask generation.

Warning:
- Do not make this mandatory unless it installs cleanly.

## Research/Technique References

### CIELAB and Delta E
Use Lab color space and Delta E because RGB distance does not match human perception well.

Technique:

```text
skin RGB → Lab
shade RGB → Lab
Delta E distance = perceptual color difference
lowest distance = closest shade
```

### Face-region sampling
Prefer stable regions:
- Left cheek.
- Right cheek.
- Forehead.
- Jawline/lower cheek.

Avoid:
- Lips.
- Nose tip highlights.
- Eyes.
- Eyebrows.
- Hairline.
- Beard.
- Background.

### Pixel filtering
Use masks first, then filter pixels:

```text
1. Collect region pixels.
2. Convert to Lab/HSV.
3. Remove darkest 20% and brightest 20% by luminance.
4. Remove extreme saturation.
5. Use median color, not mean.
6. Compare region medians to detect disagreement.
```

### Confidence scoring
Confidence should communicate uncertainty. It is better to say 72% with warnings than fake 99%.

Factors:
- Delta E distance.
- Region consistency.
- Valid pixel count.
- Image quality.
- Top 1 vs Top 2 separation.

## Best-Fit Implementation Decision

For the final assessment, the best fit is:

```text
MediaPipe landmarks + robust skin-region masks + Lab/CIEDE2000 matching + confidence scoring
```

Optional upgrade:

```text
BiSeNet face parsing for better skin-vs-hair/lip/eye segmentation
```

Do not start with BiSeNet. Add it only after the main pipeline works.
