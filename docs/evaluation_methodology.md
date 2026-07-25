# ShadeSense AI Evaluation Methodology

## Purpose

The offline evaluation harness calls the same `analyze_rgb_image` function as
the local Streamlit app. This prevents benchmark-only behavior and makes each
saved run reproducible from its manifest, catalog hash, Git commit, and seed.

The initial benchmark contains 400 images:

- 250 MST-E images balanced across the ten expert-labelled Monk Skin Tones.
- 150 adult FairFace validation images balanced across demographic, gender,
  and age metadata where the available sample permits.

MST-E is licensed for research and human-annotator training only. It must not
be used to train a machine-learning model. FairFace is CC BY 4.0. Raw images,
ZIP archives, and debug overlays remain local and are excluded from Git.

## What the benchmark measures

- Face-detection and full-pipeline success.
- Face-region lighting, extraction quality, and uncertainty.
- Readiness and recapture behavior.
- Same-subject CIEDE2000 variation relative to a front-facing, well-lit,
  unmasked reference selected for this foundation-matching task.
- Shade-family and exact-product recommendation stability.
- Results sliced by expert MST label, capture condition, and demographic
  coverage metadata.

FairFace demographic labels are never treated as skin tone. MST labels are
perceived/expert tone categories, not physical Lab measurements.

## What the benchmark does not measure

Neither public dataset contains measured physical foundation swatches or
verified wearer-to-product matches. Therefore, these runs do not establish
exact foundation-product accuracy. Claims are limited to extraction
repeatability, robustness, capture gating, and recommendation stability.

## Data separation

MST-E identities are assigned either to development or locked test. One
subject for every available MST level is held out. MST 7 and MST 10 have only
one subject in the dataset, so those tones are test-only. FairFace rows use a
stable hash split into development, validation, and locked test.

The manifest is frozen before the baseline. Later algorithm changes reuse the
same manifest. Readiness calibration may use only development data; the locked
test split is evaluated after decisions are fixed.

Images are analyzed with a maximum side of 1600 pixels using deterministic
area resampling. This preserves enough facial detail for landmark masks while
preventing high-resolution camera files from changing runtime and patch density
by an order of magnitude. Original dimensions and the applied scale are saved
per image.

## Running locally

Build the manifest:

```powershell
python scripts/build_evaluation_manifest.py `
  --dataset-root "C:\Users\abdul\Desktop\shadesense-datasets" `
  --output data/evaluation/benchmark_manifest.csv
```

Run a five-image smoke evaluation:

```powershell
python scripts/evaluate_dataset.py `
  --dataset-root "C:\Users\abdul\Desktop\shadesense-datasets" `
  --manifest data/evaluation/benchmark_manifest.csv `
  --output outputs/evaluation/smoke `
  --max-images 5 `
  --save-overlays
```

Run or resume the complete baseline:

```powershell
python scripts/evaluate_dataset.py `
  --dataset-root "C:\Users\abdul\Desktop\shadesense-datasets" `
  --manifest data/evaluation/benchmark_manifest.csv `
  --output outputs/evaluation/baseline-v1 `
  --run-label baseline-v1 `
  --resume `
  --save-overlays
```

Each run saves per-image CSV/JSONL records, Top 3 recommendations, aggregate
and subgroup metrics, repeatability results, charts, Markdown, and a standalone
HTML report.
