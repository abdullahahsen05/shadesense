# 02 — Technical Architecture

## Local-Only Scope
The current implementation target is a local Streamlit app named **ShadeSense AI**. Do not add deployment, Vercel, FastAPI, authentication, database, or cloud storage in this build cycle.


## System Overview

```text
Streamlit UI
    ↓
Image Loader
    ↓
Face Detection + Landmarks
    ↓
Region Mask Builder
    ↓
Lighting Correction
    ↓
Skin Pixel Filtering
    ↓
Representative Skin Color Extraction
    ↓
Catalog Loader + Catalog Color Conversion
    ↓
Shade Matcher
    ↓
Confidence + Explanation
    ↓
Visual Results
```

## Module Responsibilities

### `app.py`
Streamlit UI only.

Responsibilities:
- Upload image.
- Trigger pipeline.
- Display debug visuals.
- Display Top 3 shade recommendations.
- Show warnings/errors.

Should NOT contain:
- Landmark polygon logic.
- Delta E logic.
- Confidence formulas.
- Catalog parsing internals.

### `src/face_detection.py`
Responsibilities:
- Initialize MediaPipe.
- Detect face landmarks from RGB image.
- Return normalized and pixel-space landmarks.
- Handle no-face and multiple-face cases.

Expected functions:

```python
def detect_face_landmarks(image_rgb: np.ndarray) -> FaceDetectionResult:
    ...
```

### `src/region_masks.py`
Responsibilities:
- Convert selected facial landmarks to polygon masks.
- Create masks for left cheek, right cheek, forehead, jawline.
- Exclude unsafe areas where possible.

Expected functions:

```python
def build_region_masks(image_shape, landmarks) -> dict[str, np.ndarray]:
    ...
```

### `src/color_correction.py`
Responsibilities:
- Mild gray-world white balance.
- Optional gamma correction.
- Optional CLAHE on luminance.
- Return corrected image and notes.

Expected functions:

```python
def apply_mild_color_correction(image_rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    ...
```

### `src/skin_extraction.py`
Responsibilities:
- Collect pixels from region masks.
- Filter shadows/highlights using luminance percentiles.
- Filter extreme saturation/redness where needed.
- Compute median RGB and median Lab.
- Compute per-region quality scores.

Expected functions:

```python
def extract_skin_tone(image_rgb: np.ndarray, masks: dict[str, np.ndarray]) -> SkinToneResult:
    ...
```

### `src/shade_catalog.py`
Responsibilities:
- Load catalog from CSV.
- Normalize HEX/RGB values.
- Convert shade colors to Lab.
- Validate required columns.

Expected functions:

```python
def load_shade_catalog(path: str) -> pd.DataFrame:
    ...
```

### `src/shade_matcher.py`
Responsibilities:
- Compute Delta E distance between extracted skin color and each catalog shade.
- Sort shades by lowest distance.
- Return Top 3.

Expected functions:

```python
def match_shades(
    skin_lab: np.ndarray,
    catalog_df: pd.DataFrame,
    top_k: int = 3,
    uncertainty_labs=None,
    lighting_sensitivity_labs=None,
) -> list[ShadeMatch]:
    ...
```

### `src/confidence.py`
Responsibilities:
- Convert distribution-aware Delta E into a candidate color-fit score.
- Prefer Top-3 shade-family stability across bootstrap and lighting variations.
- Mark exact-product stability explicitly when it must be used as a fallback.
- Combine only available candidate evidence and normalize the remaining weights.
- Apply the global capture-readiness ceiling without flattening candidates.

Expected functions:

```python
def compute_confidence(matches, quality_report, readiness=None) -> list[ShadeMatch]:
    ...
```

### `src/explanation.py`
Responsibilities:
- Create deterministic natural-language explanations.
- Mention relevant factors: closest Lab distance, undertone, depth, quality warning.

Expected functions:

```python
def build_explanation(match: ShadeMatch, skin_result: SkinToneResult, quality_report) -> str:
    ...
```

### `src/visualization.py`
Responsibilities:
- Draw landmarks.
- Draw region masks.
- Create skin swatch image.
- Create shade swatch cards.

## Data Classes
Use dataclasses for clarity.

```python
@dataclass
class FaceDetectionResult:
    success: bool
    landmarks: list | None
    face_count: int
    warnings: list[str]

@dataclass
class SkinToneResult:
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    region_results: dict
    quality_score: float
    bootstrap_labs: list[tuple[float, float, float]]
    uncertainty_diagnostics: dict
    warnings: list[str]

@dataclass
class ShadeMatch:
    shade_id: str
    brand: str
    shade_name: str
    hex: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    delta_e: float
    confidence: float
    candidate_confidence: float
    color_fit_score: float
    shade_family_stability_score: float | None
    confidence_stability_source: str
    recommendation_stability: float
    top3_stability: float
    catalog_quality_score: float
    explanation: str
```

## Confidence Model
Confidence deliberately separates one session-level property from three
candidate-level properties.

Capture readiness measures whether the photo and extracted tone are usable. It
absorbs lighting, face quality, region agreement, valid-pixel coverage, pose,
eyewear, and extraction uncertainty. It sets the maximum candidate score:

| Readiness | Maximum candidate confidence |
|---|---:|
| Ready | 93% |
| Caution | 75% |
| Provisional | 55% |

Color fit uses the uncertainty-aware candidate distance:

```text
color_fit = exp(-distribution_aware_delta_e / 15)
```

| Distribution-aware Delta E | Color fit |
|---:|---:|
| 0.4 | 97% |
| 1 | 94% |
| 2 | 88% |
| 5 | 72% |
| 10 | 51% |

Candidate stability prefers Top-3 shade-family stability so near-equivalent
SKUs do not look unstable merely because they exchange positions:

```text
candidate_stability =
    normalized(70% bootstrap Top-3 family stability
             + 30% lighting Top-3 family stability)
```

If family stability is unavailable, the same combination of exact-product
Top-3 signals is used and marked `exact_product_fallback`. Missing factors are
omitted; they are never silently treated as zero.

```text
candidate_evidence =
    normalized_available(
        65% color_fit
      + 25% candidate_stability
      + 10% catalog_evidence
    )

candidate_confidence = readiness_cap * candidate_evidence
```

Raw CIEDE2000 remains the primary ranking signal. Candidate confidence describes
the strength of evidence for each displayed item, so a lower-ranked shade may
show slightly higher confidence when its shade family is materially more stable.
These scores are transparent engineering heuristics, not calibrated
probabilities of real-world shade satisfaction.

## Matching Algorithm
Use Lab + CIEDE2000 if available through `skimage.color.deltaE_ciede2000`.

Fallback:
- Use CIE76 Euclidean distance in Lab only if CIEDE2000 causes issues.
- Never use raw RGB Euclidean distance as the main scoring method.

## Failure Handling
The app must not crash.

Handle:
- No face detected.
- Multiple faces detected.
- Image too small.
- Catalog file missing.
- Invalid HEX/RGB values.
- Less than 3 shades in catalog.
- Too few valid skin pixels.
