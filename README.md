# ShadeSense AI

A local-first computer-vision app that analyzes a facial image and recommends the
Top 3 most suitable cosmetic foundation shades from a shade catalog, with confidence
scores and short explanations.

## Status

Local working app only. No deployment, backend service, database, or auth in this build.

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

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Usage

1. Upload a facial photo (jpg/png/bmp).
2. The app detects the face and landmarks.
3. Cheek, forehead, and jawline masks are extracted.
4. A representative skin color is computed from filtered skin pixels.
5. The skin color is matched against the selected local catalog using CIEDE2000
   perceptual color distance in Lab space.
6. The app shows the Top 3 shade recommendations with confidence scores and reasoning.

## Project Structure

```text
shadesense-ai/
├── app.py                 # Streamlit UI only
├── requirements.txt
├── data/
│   ├── shade_catalog_mock.csv
│   └── sample_images/
├── src/                   # CV / matching logic (no Streamlit imports)
│   ├── config.py
│   ├── face_detection.py
│   ├── region_masks.py
│   ├── color_correction.py
│   ├── skin_extraction.py
│   ├── shade_catalog.py
│   ├── shade_matcher.py
│   ├── confidence.py
│   ├── explanation.py
│   └── visualization.py
├── tests/
└── outputs/debug/
```

## Shade Catalog

`data/public_shade_catalog.csv` is the default catalog when present and valid.
It is generated locally from downloaded public CSV files placed in
`data/public_catalog_raw/`:

```bash
python scripts/prepare_public_catalog.py
```

`data/shade_catalog_mock.csv` remains available as a small development fallback
catalog (18 shades). The app lets you choose between the public and mock
catalogs.

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

## Known Limitations

See `docs/limitations.md` (added as phases progress) for current pipeline limitations.
