# ShadeSense AI evaluation — v3-final-master-2026-07-26

## Reproducibility

- Git commit: `2186019b84fc1948b80456aa67bf7b35bd1bd38d`
- Manifest SHA-256: `386cb41eef958536b36ce3ed987cd87cdde3ea117c258fb0b4b5177dc2b72717`
- Catalog SHA-256: `8bc98cfe9e1577985a40784cb01e31dc098168304b2c1c5c500aa5fe0467d23a`
- Random seed: `unknown`
- Images requested: 400

## Core results

- Face detection rate: 92.2%
- Full pipeline success rate: 92.2%
- Median extraction quality: 68.7/100
- Median total capture uncertainty: 6.66 Delta E
- 90th-percentile total capture uncertainty: 15.43 Delta E
- Median same-subject shift from usable MST-E reference images: 19.61 Delta E
- 90th-percentile same-subject shift: 42.52 Delta E

## Capture gating against clear MST-E metadata labels

- Clearly labelled captures: 157
- Usable-capture acceptance rate: 49.1%
- Usable-capture Ready rate: 1.8%
- Recapture rejection rate: 50.0%
- Recapture false-usable rate: 50.0%
- Dangerous false-ready rate: 2.9%

These readiness labels are metadata-derived proxies, not foundation shade ground truth. Exact product accuracy is not claimed because the public datasets do not provide verified physical foundation matches.

## Readiness distribution

- provisional: 181
- caution: 163
- unavailable: 31
- ready: 25

## Top-ranked product types

- foundation: 320
- powder: 26
- stick: 23

## Failures

- Images without a complete result: 31
- `mste-0032` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0034` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0039` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0055` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0064` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0084` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0118` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0134` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0152` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0158` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0160` (mste): No face detected. Please upload a clear, front-facing photo.
- `mste-0162` (mste): No face detected. Please upload a clear, front-facing photo.

## Subgroup audit

Full subgroup metrics are saved in `subgroup_metrics.csv`. Race/demographic labels are used only to audit system coverage and are never treated as skin-tone ground truth.
