# ShadeSense AI

ShadeSense AI is a local-first computer-vision app for foundation shade
recommendation. It analyzes a facial image, extracts a robust skin-tone estimate
from cheek, forehead, and jawline regions, and recommends the Top 3 closest
foundation shades from a local catalog with match confidence and explanations.

## Status

Local working app only. No deployment, backend service, database, or auth in this
build.

## Requirements

- Python 3.10+ (tested on 3.12)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python scripts/download_model.py   # fetches MediaPipe's face_landmarker.task (~3.7MB, one-time)
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, usually `http://localhost:8501`.

## Usage

1. Upload one facial photo, or two to three independent photos for consensus.
2. The app detects the face and MediaPipe face landmarks.
3. Cheek, forehead, and side-jaw masks are extracted while avoiding eyes, lips,
   eyebrows, hairline, central chin, and under-chin shadow where possible.
4. Face-aware lighting diagnostics measure the actual forehead, cheek, and
   side-jaw regions instead of the image background.
5. A representative skin color is computed from adaptive diffuse patches using
   a CIEDE2000 medoid, robust outlier rejection, and bounded region influence.
6. Deterministic patch bootstrapping quantifies extraction uncertainty; central,
   median, and 90th-percentile CIEDE2000 evidence all influence ranking.
7. The app separates measured visible skin tone from foundation target tone when
   glossy highlights could bias rich/deep skin recommendations too light.
8. Six conservative exposure, white-balance, and gamma perturbations test whether
   extraction and shade rankings remain stable under plausible capture variation.
9. Capture-level uncertainty separately models global exposure, asymmetry,
   color-cast, pose, and eyewear risks that patch bootstrap cannot observe.
10. The target color is matched against the selected local catalog using CIEDE2000
   perceptual color distance in Lab space, with metadata affecting only close ties.
11. Readiness incorporates post-match exact-SKU and shade-family stability.
    Perceptually near-equivalent catalog colors are grouped so distinct shade
    families appear before duplicate-looking products in the Top 3.
12. With multiple photos, each capture is analyzed independently; a weighted
    CIEDE2000 medoid rejects a gross whole-photo outlier and anchors consensus
    to a real retained observation.
13. The app always shows visually distinct Top 3 candidates with readiness-aware
    confidence, uncertainty diagnostics, and reasoning.

## Project Structure

```text
shadesense-ai/
app.py                 # Streamlit UI only
requirements.txt
data/
  shade_catalog_mock.csv
  public_shade_catalog.csv
  sample_images/
src/                   # CV and matching logic
  face_detection.py
  region_masks.py
  color_correction.py
  lighting_quality.py
  image_quality.py
  skin_extraction.py
  extraction_quality.py
  capture_uncertainty.py
  multicapture_consensus.py
  multi_photo_consensus.py
  shade_catalog.py
  shade_matcher.py
  confidence.py
  explanation.py
  visualization.py
scripts/
  evaluate_dataset.py
  evaluate_multi_photo.py
  evaluate_repeatability.py
  mask_audit_app.py
  prepare_public_catalog.py
tests/
docs/
```

## Core Pipeline

```text
image upload
-> embedded ICC conversion to sRGB
-> lighting and capture-quality diagnostics
-> MediaPipe face detection and landmarks
-> cheek / forehead / side-jaw masks
-> adaptive skin-pixel filtering
-> adaptive patch extraction and perceptual medoid consensus
-> deterministic bootstrap uncertainty
-> systematic capture uncertainty
-> optional depth-safe foundation target adjustment
-> CIEDE2000 shade matching with uncertainty and catalog evidence
-> post-match shade-family stability and readiness
-> Top 3 recommendations, readiness-aware confidence, and explanations
```

## Multi-Capture Repeatability

The Streamlit app accepts up to three face photos. Each photo runs through the
complete pipeline independently. The consensus layer quality-weights the
captures, chooses a real observed CIEDE2000 medoid, rejects a gross outlier only
when there is enough evidence, and caps confidence when two captures disagree.

For a small personal-photo repeatability check:

```bash
python scripts/evaluate_repeatability.py "C:\path\to\capture-folder" --pattern "Image_*.jpeg"
```

The evaluator does not copy or commit the photographs. It reports per-capture
Lab estimates, low-signal/recapture flags, combined uncertainty, a weighted
CIEDE2000 medoid, whole-photo outliers, and between-capture repeatability. The
consensus uses an actual retained observation rather than synthesizing a skin
color between incompatible captures.

For the frozen MST-E benchmark, create held-out-reference multi-photo evidence
after the main evaluation finishes:

```bash
python scripts/evaluate_multi_photo.py \
  --results outputs/evaluation/baseline-v1/per_image_results.csv \
  --output outputs/evaluation/baseline-v1/multi_photo
```

The designated reference photo is excluded from consensus inputs and is used
only to measure repeatability.

## Dataset Evaluation

The frozen 400-image manifest contains 250 MST-E and 150 adult FairFace
validation images. The harness calls the same shared analysis function as the
app and saves restartable records, subgroup metrics, charts, Markdown, and HTML.
See `docs/evaluation_methodology.md` for dataset licenses, split rules, commands,
and the strict boundary between robustness evidence and physical product-match
accuracy. The completed baseline/candidate measurements and presentation
artifact index are recorded in `docs/evaluation_results.md`.

## Shade Catalog

`data/public_shade_catalog.csv` is the default catalog when present and valid.
It is generated locally from downloaded public CSV files placed in
`data/public_catalog_raw/`:

```bash
python scripts/prepare_public_catalog.py
```

`data/shade_catalog_mock.csv` remains available as a small development fallback
catalog. The app lets you choose between the public and mock catalogs.

Mock catalog schema:

```csv
shade_id,brand,shade_name,hex,r,g,b,undertone,depth,notes
```

Public catalog schema:

```csv
shade_id,brand,product,shade_name,hex,undertone,depth,source,source_url
```

See `docs/catalog_setup.md` for setup details and limitations. Public swatch
colors are website-derived approximations, not guaranteed matches to real
applied foundation.

The app does not scrape Sephora, Kaggle, or any live website at runtime.

## Tests

```bash
pytest tests/ -v
python -m compileall src scripts app.py
```

## Known Limitations

See `docs/limitations.md` for current pipeline limitations.
