# ShadeSense AI Evaluation Results

## What this evaluation establishes

ShadeSense AI was evaluated on a frozen, stratified 400-image manifest:

- 250 MST-E images, balanced across the 10 Monk Skin Tone groups
- 150 adult FairFace validation images
- manifest SHA-256:
  `386cb41eef958536b36ce3ed987cd87cdde3ea117c258fb0b4b5177dc2b72717`

The harness calls the same analysis pipeline as the Streamlit app. It records
face detection, extraction quality, capture uncertainty, readiness, product
type, subgroup behavior, repeatability, and debug overlays. Results are saved
under `outputs/evaluation/`, which is intentionally ignored by Git because it
contains generated artifacts and potentially sensitive derived image outputs.

This benchmark measures capture robustness and recommendation consistency. It
does **not** establish physical foundation accuracy because neither dataset
contains verified wearer-to-product matches or measured applied foundation
swatches.

## Reproducible runs

| Run | Analysis commit | Images | Purpose |
|---|---|---:|---|
| `baseline-v1` | `75cabefdd3396cdb3327ae84abd5fe4212ca1192` | 400 | Frozen pre-improvement baseline |
| `v2-color-catalog` | `d6b80ae0c6a335175ef86b0f1af994c71e2dd0aa` | 400 | Face fallback, tone-safe low-signal logic, and foundation-only catalog |

The full run configuration, manifest hash, timing, and analysis commit are also
stored inside each run directory. The candidate run was processed with
restartable parallel shards and then finalized into one deterministic report.

## Baseline versus candidate

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Face detection / pipeline success | 89.25% | 92.25% | +3.00 pp |
| Successful images | 357 | 369 | +12 |
| Median extraction quality | 65.90 | 67.05 | +1.15 |
| Median capture uncertainty | 7.06 Delta E | 6.49 Delta E | -0.57 |
| P90 capture uncertainty | 13.18 Delta E | 12.71 Delta E | -0.47 |
| Median repeatability distance | 18.02 Delta E | 18.04 Delta E | +0.02 |
| P90 repeatability distance | 44.66 Delta E | 44.51 Delta E | -0.15 |

The paired comparison contains all 400 manifest rows, including failed
detections. Twelve images changed from failed to successful analysis. Median
matching-Lab change on images successful in both runs was zero, which is
expected: the improvements primarily changed failure handling, uncertainty,
readiness, and eligible product scope rather than silently shifting already
stable skin estimates.

Top-1 changed on 12.25% of paired images. This is mainly catalog-scope behavior
and must not be presented as a 12.25% accuracy gain.

## Skin-tone subgroup audit

Every MST group contains 25 images. Candidate face detection was equal or
better in every group, and median capture uncertainty decreased in every group.

| MST | Candidate detection | Detection change | Candidate median uncertainty | Uncertainty change |
|---:|---:|---:|---:|---:|
| 1 | 100% | 0 pp | 5.79 | -0.11 |
| 2 | 88% | +8 pp | 7.18 | -0.74 |
| 3 | 92% | 0 pp | 7.95 | -0.52 |
| 4 | 96% | 0 pp | 6.99 | -1.08 |
| 5 | 96% | 0 pp | 6.76 | -1.44 |
| 6 | 96% | +4 pp | 7.06 | -0.78 |
| 7 | 76% | +4 pp | 7.50 | -1.25 |
| 8 | 92% | +4 pp | 6.49 | -0.08 |
| 9 | 88% | +4 pp | 6.92 | -0.25 |
| 10 | 96% | 0 pp | 5.71 | -0.38 |

MST 7 remains the weakest detection subgroup at 76%. That is a concrete target
for future landmark/mask auditing rather than evidence for a broad model
replacement.

## Readiness calibration

Readiness thresholds were selected only on the MST-E development partition and
then audited once on a locked partition. Labels such as lighting, pose, and
mask quality are capture-quality proxies—not product-match ground truth.

Selected thresholds:

- Caution: readiness score at least 52, uncertainty at most 9 Delta E, and
  lighting sensitivity at most 5.5 Delta E
- Ready: readiness score at least 78, uncertainty at most 7 Delta E, and
  lighting sensitivity at most 4 Delta E
- Existing extraction-quality and recommendation-stability gates remain active

| Calibration evidence | Usable accepted | Recapture falsely usable | Dangerous false Ready |
|---|---:|---:|---:|
| Development selection | 56.52% | 38.64% | 2.27% |
| Locked audit | 40.63% | 27.59% | 0.00% |

The candidate run's generated `aggregate_metrics.json` contains the readiness
state produced before these final thresholds were selected. The committed
`data/evaluation/readiness_calibration.json` is the authoritative configuration
used by the app after calibration.

## Foundation-only recommendation scope

The candidate's Top-1 product types were:

- foundation: 315
- powder: 30
- stick: 24

The baseline also returned BB/CC, tint, tinted moisturizer, cushion,
concealer-hybrid, and generic base products. Those adjacent categories no
longer enter the default foundation-only recommendation set. Users can still
select broader catalog modes explicitly.

This improves category relevance. It does not calibrate website-derived catalog
colors to the physical appearance of applied products.

## Multi-photo consensus

The held-out-reference evaluator groups independent captures by identity. The
designated usable reference is never included in consensus and is used only for
repeatability scoring.

| Metric | Baseline | Candidate |
|---|---:|---:|
| Subjects | 19 | 19 |
| Median consensus-to-reference | 19.38 Delta E | 13.76 Delta E |
| P90 consensus-to-reference | 46.65 Delta E | 40.40 Delta E |
| Subjects improved | 42.11% | 47.37% |
| Paired median improvement | 0.00 Delta E | 0.00 Delta E |

The aggregate median improved, but fewer than half of identities improved and
the paired median improvement was zero. Multi-photo consensus is therefore a
useful conservative option, not a universal accuracy guarantee. The high
capture-exclusion rate also shows that the available grouped captures often
disagree substantially.

## Mask audit status

The evaluation generated a deterministic 100-image manual audit queue with
region overlays. Human labels are still required before deciding whether a
semantic face-parsing model is justified. MediaPipe remains the only face model
until reviewed masks demonstrate repeated, material contamination by facial
hair, eyewear, lips, or background.

## Presentation artifact index

The following local artifacts contain the evidence behind this summary:

- `outputs/evaluation/baseline-v1/report.html`
- `outputs/evaluation/v2-color-catalog/report.html`
- `outputs/evaluation/baseline-v1-vs-v2/comparison_summary.md`
- `outputs/evaluation/baseline-v1-vs-v2/paired_image_comparison.csv`
- `outputs/evaluation/v2-color-catalog/readiness_calibration_summary.md`
- `outputs/evaluation/v2-color-catalog/multi_photo/multi_photo_summary.md`
- `outputs/evaluation/v2-color-catalog/mask_audit.csv`
- `outputs/evaluation/face-fallback-audit.jpg`

Each main run also contains per-image CSV/JSONL records, aggregate and subgroup
metrics, recommendation records, charts, debug overlays, and a run
configuration file.

## Claims that are and are not supported

Supported:

- the candidate completes more images than the baseline;
- capture uncertainty is lower overall and across every audited MST group;
- default recommendations now stay within foundation product categories;
- readiness thresholds were selected on development evidence and audited on a
  locked partition;
- multi-photo consensus improves aggregate held-out repeatability in this small
  identity sample.

Not supported:

- a percentage of physically correct shade matches;
- superiority to a trained shade-classification model on labeled product
  ground truth;
- calibrated confidence probabilities;
- equivalence between public website HEX colors and applied foundation;
- training or fine-tuning on MST-E or FairFace. These datasets are evaluation
  inputs only.
