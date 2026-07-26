# Candidate Confidence Validation

This focused check uses authorized self-test photos. It validates score semantics and differentiation; it does not measure correct-shade accuracy.

## Summary

- **Images Requested:** 7
- **Images Succeeded:** 7
- **Candidates:** 21
- **Images With At Least 1Pp Candidate Spread:** 7
- **Images With Rank Confidence Inversion:** 2
- **Mean Candidate Spread:** 0.0982
- **Color Fit Candidate Evidence Correlation:** 0.9417
- **Color Fit Normalized Confidence Correlation:** 0.9417
- **Mean Absolute Candidate Evidence Minus Color Fit:** 0.0568
- **Catalog Evidence Unique Values:** [0.8, 0.9, 1.0]
- **Catalog Evidence Is Constant:** false
- **Stability Source Counts:** {"shade_family": 21}
- **All Confidences Respect Readiness Cap:** true

## Per-capture outcome

| image | success | readiness_state | readiness_score | readiness_cap | candidate_spread |
| --- | --- | --- | --- | --- | --- |
| Image_01_bright-close-no-glasses.jpeg | True | caution | 63.04855203461639 | 0.6349888618047416 | 0.08116023818683549 |
| Image_02_bright-front-glasses-reflections.jpeg | True | caution | 68.03092386906862 | 0.6733147989928355 | 0.10500085630501843 |
| Image_03_dimmer-lighting-glasses.jpeg | True | provisional | 44.6803832028452 | 0.55 | 0.05283061229925229 |
| Image_04_bright-angled-no-glasses.jpeg | True | caution | 75.2971325669441 | 0.7292087120534162 | 0.10682457795185651 |
| Image_05_dim-asymmetric-lighting-glasses.jpeg | True | provisional | 42.096620881910646 | 0.55 | 0.1389718774448349 |
| Image_06_reference-bright-front-no-glasses.jpeg | True | caution | 71.26238280481557 | 0.6981721754216583 | 0.13977934423420402 |
| Image_07_bright-close-glasses.jpeg | True | caution | 58.623325657524 | 0.6009486589040308 | 0.06255671769182453 |

## Candidate evidence

| image | rank | shade_name | readiness_state | readiness_cap | distribution_delta_e | color_fit | shade_family_stability | stability_source | catalog_evidence | candidate_evidence | candidate_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image_01_bright-close-no-glasses.jpeg | 1 | 3 | caution | 63.5% | 0.72 | 95.3% | 95.0% | shade_family | 80.0% | 93.7% | 59.5% |
| Image_01_bright-close-no-glasses.jpeg | 2 | pebble | caution | 63.5% | 2.09 | 87.0% | 67.4% | shade_family | 90.0% | 82.4% | 52.3% |
| Image_01_bright-close-no-glasses.jpeg | 3 | 7 | caution | 63.5% | 2.18 | 86.5% | 58.9% | shade_family | 100.0% | 80.9% | 51.4% |
| Image_02_bright-front-glasses-reflections.jpeg | 1 | ivory beige | caution | 67.3% | 1.00 | 93.5% | 89.3% | shade_family | 100.0% | 93.1% | 62.7% |
| Image_02_bright-front-glasses-reflections.jpeg | 2 | N35 | caution | 67.3% | 1.66 | 89.6% | 79.0% | shade_family | 100.0% | 87.9% | 59.2% |
| Image_02_bright-front-glasses-reflections.jpeg | 3 | warm beige | caution | 67.3% | 2.06 | 87.1% | 43.5% | shade_family | 100.0% | 77.5% | 52.2% |
| Image_03_dimmer-lighting-glasses.jpeg | 1 | milk chocolate | provisional | 55.0% | 1.91 | 88.1% | 61.1% | shade_family | 90.0% | 81.5% | 44.8% |
| Image_03_dimmer-lighting-glasses.jpeg | 2 | deep/dark | provisional | 55.0% | 2.42 | 85.1% | 47.8% | shade_family | 90.0% | 76.3% | 42.0% |
| Image_03_dimmer-lighting-glasses.jpeg | 3 | praline | provisional | 55.0% | 2.73 | 83.4% | 35.0% | shade_family | 90.0% | 71.9% | 39.6% |
| Image_04_bright-angled-no-glasses.jpeg | 1 | light cool | caution | 72.9% | 0.85 | 94.5% | 95.0% | shade_family | 100.0% | 95.2% | 69.4% |
| Image_04_bright-angled-no-glasses.jpeg | 2 | 3 | caution | 72.9% | 1.86 | 88.3% | 87.8% | shade_family | 100.0% | 89.4% | 65.2% |
| Image_04_bright-angled-no-glasses.jpeg | 3 | 15C | caution | 72.9% | 1.89 | 88.2% | 52.9% | shade_family | 100.0% | 80.5% | 58.7% |
| Image_05_dim-asymmetric-lighting-glasses.jpeg | 1 | cacao | provisional | 55.0% | 1.67 | 89.5% | 80.0% | shade_family | 90.0% | 87.2% | 47.9% |
| Image_05_dim-asymmetric-lighting-glasses.jpeg | 2 | rich | provisional | 55.0% | 3.31 | 80.2% | 13.0% | shade_family | 90.0% | 64.4% | 35.4% |
| Image_05_dim-asymmetric-lighting-glasses.jpeg | 3 | 23 | provisional | 55.0% | 3.54 | 79.0% | 10.2% | shade_family | 80.0% | 61.9% | 34.0% |
| Image_06_reference-bright-front-no-glasses.jpeg | 1 | alabaster | caution | 69.8% | 0.82 | 94.7% | 93.5% | shade_family | 90.0% | 93.9% | 65.6% |
| Image_06_reference-bright-front-no-glasses.jpeg | 2 | bareMinerals | caution | 69.8% | 2.16 | 86.6% | 38.5% | shade_family | 80.0% | 73.9% | 51.6% |
| Image_06_reference-bright-front-no-glasses.jpeg | 3 | natural | caution | 69.8% | 2.27 | 85.9% | 64.5% | shade_family | 100.0% | 82.0% | 57.2% |
| Image_07_bright-close-glasses.jpeg | 1 | warm | caution | 60.1% | 1.25 | 92.0% | 92.8% | shade_family | 100.0% | 93.0% | 55.9% |
| Image_07_bright-close-glasses.jpeg | 2 | teak | caution | 60.1% | 2.25 | 86.1% | 70.5% | shade_family | 90.0% | 82.6% | 49.6% |
| Image_07_bright-close-glasses.jpeg | 3 | cool beige | caution | 60.1% | 2.37 | 85.4% | 69.1% | shade_family | 100.0% | 82.8% | 49.7% |

## Interpretation rules

- Ranking remains color-first; confidence is an evidence-strength score.
- A lower-ranked candidate may have higher confidence when its shade family is more stable.
- Constant catalog evidence is an intended baseline, not a rank discriminator.
- Candidate confidence must never exceed its capture-readiness cap.
- Family stability is preferred; exact-product fallback must be marked.
