# ShadeSense AI Evaluation Results

This document records the current final evaluation state. The presentation-ready
charts and machine-readable evidence are published in
[`docs/evaluation/final-400-image-run`](evaluation/final-400-image-run/README.md).

## Final Frozen Run

Run label: `v3-final-master-2026-07-26`

| Metric | Result |
|---|---:|
| Images | 400 |
| MST-E | 250, balanced at 25 per expert Monk Skin Tone |
| Adult FairFace validation | 150 |
| Face detection / pipeline success | 92.25% |
| Successful extractions | 369 |
| Median extraction quality | 68.69 / 100 |
| Median capture uncertainty | 6.66 Delta E |
| 90th-percentile capture uncertainty | 15.43 Delta E |
| Median processing time | 8.78 seconds |

Readiness counts were 25 ready, 163 caution, 181 provisional, and 31
unavailable. Failures were retained in the denominator.

## Subgroup Findings

- Every expert Monk Skin Tone group contains 25 images.
- Tone 7 had the lowest pipeline success at 76%; the remaining levels ranged
  from 88% to 100%.
- Frontal/facing-camera captures substantially outperformed side poses.
- Masked faces had lower processing coverage than unmasked faces.
- Poor lighting increased same-subject Delta E variation.

These are failure-discovery findings. They must not be restated as measured
product-shade accuracy.

## Multi-Photo Consensus

Nineteen MST-E identities had a designated held-out reference. The reference
was excluded from consensus inputs.

| Metric | Result |
|---|---:|
| Median individual-to-reference distance | 21.26 Delta E |
| Median consensus-to-reference distance | 11.24 Delta E |
| Median improvement | 5.98 Delta E |
| Subjects improved | 52.6% |

## Candidate-Confidence Validation

Seven real photographs produced 21 candidate recommendations:

- every image had candidate-specific confidence spread;
- mean Top-3 spread was 9.8 percentage points;
- all confidences respected the capture-readiness ceiling;
- all stability evidence used shade-family grouping;
- catalog evidence varied across candidates.

## Supported Claims

The evaluation supports claims about:

- face-detection and end-to-end processing coverage;
- behavior across source, skin-tone, lighting, pose, age, gender, eyewear, and
  mask subgroups;
- extraction quality and uncertainty;
- capture-readiness behavior;
- same-person repeatability;
- multi-photo consensus;
- candidate-specific confidence semantics.

It does not support claims about:

- physical foundation wear-test accuracy;
- calibrated confidence probabilities;
- device-independent color accuracy;
- clinical or dermatological validity;
- training or fine-tuning on MST-E or FairFace.

Neither dataset supplies verified matches from the exact product catalog.
Public catalog colors are website-derived approximations, not measured applied
foundation swatches.
