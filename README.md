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

1. Upload a facial photo, such as JPG, PNG, or BMP.
2. The app detects the face and MediaPipe face landmarks.
3. Cheek, forehead, and jawline masks are extracted while avoiding eyes, lips,
   eyebrows, hairline, and under-chin shadow where possible.
4. Face-aware lighting diagnostics measure the actual forehead, cheek, and
   jawline regions instead of the image background.
5. A representative skin color is computed from adaptive diffuse patches using
   a CIEDE2000 medoid, robust outlier rejection, and bounded region influence.
6. Deterministic patch bootstrapping quantifies extraction and recommendation
   stability.
7. The app separates measured visible skin tone from foundation target tone when
   glossy highlights could bias rich/deep skin recommendations too light.
8. The target color is matched against the selected local catalog using CIEDE2000
   perceptual color distance in Lab space, with metadata affecting only close ties.
9. The app always shows visually distinct Top 3 candidates with readiness-aware
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
  shade_catalog.py
  shade_matcher.py
  confidence.py
  explanation.py
  visualization.py
scripts/
  prepare_public_catalog.py
tests/
docs/
```

## Core Pipeline

```text
image upload
-> lighting and capture-quality diagnostics
-> MediaPipe face detection and landmarks
-> cheek / forehead / jawline masks
-> adaptive skin-pixel filtering
-> adaptive patch extraction and perceptual medoid consensus
-> deterministic bootstrap uncertainty
-> optional depth-safe foundation target adjustment
-> CIEDE2000 shade matching with uncertainty and catalog evidence
-> Top 3 recommendations, readiness-aware confidence, and explanations
```

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
