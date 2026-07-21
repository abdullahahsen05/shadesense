# Phase 02 — Cheek, Forehead, and Jawline Masks

## Goal
Identify primary skin regions: cheeks, forehead, and jawline.

## Tasks
1. Implement `src/region_masks.py`.
2. Define landmark-index groups for:
   - Left cheek.
   - Right cheek.
   - Forehead.
   - Jawline/lower cheek.
3. Convert each group to polygon masks using OpenCV.
4. Add mask visualization.
5. Show each region separately in Streamlit.
6. Add combined skin-region mask.

## Mask Quality Rules
Masks should avoid:
- Eyes.
- Lips.
- Eyebrows.
- Nose tip highlight where possible.
- Hairline.
- Background.

## Acceptance Criteria
- App shows separate masks for cheeks, forehead, and jawline.
- Masks align reasonably on at least 3 different sample faces.
- Regions do not obviously include eyes/lips/background.
- App does not crash if a polygon has insufficient points.

## Do Not Do Yet
- Do not over-optimize landmark indices forever.
- Get reasonable masks first, then improve in Phase 03.


## Commit and Continue Rule
After this phase passes its acceptance criteria:

```bash
git status
git add .
git commit -m "phase-02: add facial region masks"
```

Then continue automatically to the next phase. Do not ask for permission to proceed unless a true blocker appears. Do not push to GitHub.
