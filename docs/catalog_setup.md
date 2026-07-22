# Public Catalog Setup

ShadeSense AI can use a downloaded public Sephora-style foundation swatch
dataset as a local catalog. The app does not scrape Sephora, Kaggle, or any live
website at runtime.

## Place Raw Files

Download the dataset outside the app, then place the raw CSV files here:

```text
data/public_catalog_raw/
```

For the current dataset zip, the detected CSV files are:

```text
allCategories.csv
allNumbers.csv
allShades.csv
sephora.csv
ulta.csv
```

## Prepare Normalized Catalog

Run the importer from the project root:

```bash
python scripts/prepare_public_catalog.py
```

The script reads every CSV in `data/public_catalog_raw/`, validates HEX swatch
colors, infers conservative undertone/depth metadata, removes duplicate
brand/product/shade/color rows, and writes:

```text
data/public_shade_catalog.csv
```

Normalized schema:

```csv
shade_id,brand,product,shade_name,hex,undertone,depth,source,source_url
```

Rows without a valid HEX color are skipped with warnings instead of crashing.
Files that do not contain a HEX-like column are skipped because the shade matcher
needs swatch color data.

## Run The App

```bash
streamlit run app.py
```

If `data/public_shade_catalog.csv` exists and contains valid rows, the app
defaults to the public catalog. Otherwise it falls back to
`data/shade_catalog_mock.csv`.

## Limitations

Public catalog colors are website-derived swatch approximations. They may differ
from real applied foundation due to lighting, display calibration, brand image
processing, oxidation, formula finish, opacity, and skin texture.

The catalog preparation is local-only. It does not download from Kaggle, scrape
Sephora or Ulta, or add any runtime network dependency.
