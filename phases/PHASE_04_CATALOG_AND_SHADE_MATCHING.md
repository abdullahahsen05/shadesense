# Phase 04 — Catalog Loading and Shade Matching

## Goal
Load the shade catalog and return Top 3 closest recommendations.

## Tasks
1. Implement `src/shade_catalog.py`.
2. Load `data/shade_catalog_mock.csv`.
3. Normalize colors from HEX/RGB.
4. Convert shade colors to Lab.
5. Implement `src/shade_matcher.py`.
6. Compute Delta E / CIEDE2000 between skin Lab and shade Lab.
7. Sort shades by ascending distance.
8. Return Top 3.
9. Display Top 3 shade cards in Streamlit.

## Matching Rules
- Main ranking: CIEDE2000 Delta E.
- Fallback: Lab Euclidean distance only if CIEDE2000 is unavailable.
- Do not use raw RGB distance as final ranking.

## Acceptance Criteria
- Catalog loads successfully.
- App shows Top 3 shade recommendations.
- Each shade card shows name, HEX/swatch, distance, and rank.
- If catalog has invalid colors, the app reports a validation error.
- If catalog has fewer than 3 rows, app shows available rows without crashing.

## Do Not Do Yet
- Do not build complicated confidence before matching works.


## Commit and Continue Rule
After this phase passes its acceptance criteria:

```bash
git status
git add .
git commit -m "phase-04: add catalog loading and shade matching"
```

Then continue automatically to the next phase. Do not ask for permission to proceed unless a true blocker appears. Do not push to GitHub.
