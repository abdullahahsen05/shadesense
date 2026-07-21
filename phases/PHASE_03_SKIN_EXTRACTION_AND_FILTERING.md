# Phase 03 — Skin Extraction and Filtering

## Goal
Extract a reliable representative skin color from masked face regions.

## Tasks
1. Implement `src/color_correction.py` with mild gray-world white balance.
2. Implement `src/skin_extraction.py`.
3. For each region:
   - Collect RGB pixels from the mask.
   - Convert pixels to Lab and/or HSV.
   - Remove darkest and brightest luminance percentiles.
   - Remove extreme saturation pixels.
   - Compute median RGB and median Lab.
   - Count valid pixels.
4. Combine reliable regions into one final skin color.
5. Compute region consistency.
6. Show extracted skin swatch in Streamlit.
7. Show warnings for poor extraction.

## Suggested Filtering
Start simple:

```text
luminance lower bound = 20th percentile
luminance upper bound = 80th percentile
remove pixels with extreme saturation
minimum valid pixels per region = 100
```

Adjust after visual testing.

## Acceptance Criteria
- Extracted swatch looks close to visible skin tone.
- Shadows/highlights do not dominate the result.
- Median color is used instead of full mean.
- App shows per-region colors and final combined color.
- Warnings appear for low valid pixel count or high region disagreement.

## Do Not Do Yet
- Do not match shades until extraction is visually acceptable.


## Commit and Continue Rule
After this phase passes its acceptance criteria:

```bash
git status
git add .
git commit -m "phase-03: add skin extraction and filtering"
```

Then continue automatically to the next phase. Do not ask for permission to proceed unless a true blocker appears. Do not push to GitHub.
