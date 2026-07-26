# Final 400-Image Evaluation Run

This directory is the GitHub-safe evidence package for the frozen
`v3-final-master-2026-07-26` evaluation. It contains aggregate results and
charts, not source photographs.

## Run Scope

- 400 images on a deterministic manifest
- 250 MST-E images: 25 per expert Monk Skin Tone level
- 150 adult FairFace validation images
- exact same `analyze_rgb_image` pipeline used by Streamlit
- no training or fine-tuning on either evaluation dataset
- failures retained in the denominator

## Headline Results

| Metric | Result |
|---|---:|
| Face detection rate | 92.25% |
| Pipeline success rate | 92.25% |
| Successful extractions | 369 / 400 |
| Median extraction quality | 68.69 / 100 |
| Median capture uncertainty | 6.66 Delta E |
| 90th-percentile capture uncertainty | 15.43 Delta E |
| Median runtime | 8.78 seconds per image |
| Readiness | 25 ready, 163 caution, 181 provisional, 31 unavailable |

## Visual Evidence

### Coverage by Expert Monk Skin Tone

![MST-E success rates by expert Monk tone](mste_success_by_tone.png)

Each expert Monk Skin Tone group contains 25 images. The weakest face-detection
group was tone 7 at 76%; all other groups were between 88% and 100%. This is a
failure-discovery result, not a claim of equal product-match accuracy.

### Readiness Distribution

![Readiness distribution](readiness_distribution.png)

The system deliberately withholds `ready` status from uncertain captures.
`Caution` and `provisional` results still return three candidates, but their
confidence is capped and the UI recommends recapture when appropriate.

### Lighting and Same-Subject Repeatability

![Repeatability by lighting](repeatability_by_lighting.png)

The 19 MST-E identities with a designated reference show that poor lighting
increases same-person Lab variation. The behavior is exposed in the report and
fed into capture uncertainty rather than hidden.

## Multi-Photo Result

The held-out-reference multi-photo evaluation used each subject's reference
only for scoring:

- median individual-to-reference distance: 21.26 Delta E;
- median consensus-to-reference distance: 11.24 Delta E;
- median improvement: 5.98 Delta E;
- 52.6% of the 19 subjects improved.

## Included Files

| File | Purpose |
|---|---|
| `aggregate_metrics.json` | Machine-readable headline metrics |
| `subgroup_metrics.csv` | Dataset, skin-tone, pose, lighting, gender, age, eyewear, and mask subgroups |
| `multi_photo_metrics.json` | Held-out multi-capture repeatability details |
| `mste_success_by_tone.png` | Skin-tone coverage chart |
| `readiness_distribution.png` | Safety/readiness chart |
| `repeatability_by_lighting.png` | Lighting sensitivity chart |

## Interpretation Boundary

These results measure face detection, usable extraction, uncertainty,
readiness, subgroup behavior, and same-subject repeatability. They do not
measure physical shade-match accuracy because neither source dataset supplies a
verified foundation match from the project's exact catalog.

The local run produced additional per-image CSV/JSON records, an HTML explorer,
and 369 face-overlay images. They are kept out of Git because the overlays
contain dataset faces and redistribution may be restricted by source-dataset
licenses.

