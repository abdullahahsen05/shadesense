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

Run the foundation-only, ICC-aware candidate after the unchanged baseline:

```powershell
python scripts/evaluate_dataset.py `
  --dataset-root "C:\Users\abdul\Desktop\shadesense-datasets" `
  --manifest data/evaluation/benchmark_manifest.csv `
  --output outputs/evaluation/v2-color-catalog `
  --run-label v2-color-catalog `
  --product-scope foundation_only `
  --resume `
  --save-overlays
```

For a faster local run on a multi-core machine, the deterministic shard wrapper
uses the same evaluator in separate processes and merges rows back into frozen
manifest order:

```powershell
python scripts/evaluate_dataset_parallel.py `
  --dataset-root "C:\Users\abdul\Desktop\shadesense-datasets" `
  --manifest data/evaluation/benchmark_manifest.csv `
  --output outputs/evaluation/v2-color-catalog `
  --run-label v2-color-catalog `
  --product-scope foundation_only `
  --workers 3 `
  --resume `
  --save-overlays
```

Each worker is independently restartable. The merged run records the parent
manifest and catalog hashes, Git commit, process count, and elapsed time.

## Readiness calibration

Readiness thresholds are selected from clear MST-E development labels only.
The optimization penalizes a false usable/ready decision on a recapture image
four times more than rejecting a usable image, subject to a minimum usable
acceptance rate. Locked-test metrics are reported only after the threshold is
selected.

```powershell
python scripts/calibrate_readiness.py `
  --results outputs/evaluation/v2-color-catalog/per_image_results.csv `
  --run-config outputs/evaluation/v2-color-catalog/run_config.json `
  --output data/evaluation/readiness_calibration.json
```

MST-E metadata provides capture-quality proxy labels, not foundation shade
ground truth. Calibration therefore improves recapture safety, not proof of
physical product accuracy. Calibration must use the improved candidate run
because transplanting thresholds from a pipeline with different low-signal
behavior would introduce distribution shift.

## Failure-driven mask audit

The audit tool selects 100 images: 60 MST-E and 40 FairFace, split between
high-risk failures and stratified representative cases. A human reviewer marks
forehead, left cheek, right cheek, and jawline as clean, minor contamination,
major contamination, or insufficient visible skin.

```powershell
python scripts/prepare_mask_audit.py `
  --results outputs/evaluation/baseline-v1/per_image_results.csv `
  --output outputs/evaluation/baseline-v1/mask_audit.csv

streamlit run scripts/mask_audit_app.py -- `
  --dataset-root "C:\Users\abdul\Desktop\shadesense-datasets" `
  --audit outputs/evaluation/baseline-v1/mask_audit.csv
```

Semantic face parsing should be added only if this review finds repeated,
material contamination that the current MediaPipe masks cannot address.

## Multi-photo evaluation

For each MST-E identity with enough captures, the designated usable reference
is held out. Two or three non-reference captures form the consensus and the
reference is used only for CIEDE2000 repeatability scoring.

```powershell
python scripts/evaluate_multi_photo.py `
  --results outputs/evaluation/v2-color-catalog/per_image_results.csv `
  --output outputs/evaluation/v2-color-catalog/multi_photo
```

This produces CSV, JSON, and Markdown artifacts showing consensus-to-reference
Delta E, individual-photo median Delta E, improvement rate, and outlier
rejection rate.

## Paired baseline comparison

After both complete runs use the same frozen manifest, write a paired artifact
bundle rather than comparing rounded headline numbers by eye:

```powershell
python scripts/compare_evaluation_runs.py `
  --baseline outputs/evaluation/baseline-v1 `
  --candidate outputs/evaluation/v2-color-catalog `
  --output outputs/evaluation/baseline-v1-vs-v2
```

The comparison reports pipeline changes, readiness transitions, extracted Lab
movement, and catalog Top-1 changes for the same image IDs. A product change is
not labeled an accuracy improvement without physical product ground truth.

## Neutral-card mode

The optional app mode estimates bounded RGB channel gains from the evenly lit
centre of a neutral gray-card reference taken with the same camera and light.
The mode rejects dark, clipped, or highly nonuniform references. Public
benchmarks do not include calibration cards, so this feature is verified with
synthetic regression tests and must not be presented as dataset-measured gain.
