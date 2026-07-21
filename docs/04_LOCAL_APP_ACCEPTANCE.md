# 04 — Local App Acceptance Checklist

## Purpose
This checklist defines what "working local app" means for the current build. It replaces any demo/submission phase for now.

## Must Work Locally
The local Streamlit app must:

1. Start with:

```bash
streamlit run app.py
```

2. Allow the user to upload a facial image.
3. Detect a face automatically.
4. Show detected face landmarks or an equivalent overlay.
5. Show cheek, forehead, and jawline region masks.
6. Extract a representative skin color from valid skin pixels.
7. Load the mock shade catalog from `data/shade_catalog_mock.csv`.
8. Return the Top 3 foundation shades.
9. Show a confidence score for each recommendation.
10. Show short reasoning for each recommendation.
11. Handle no-face and poor-quality images without crashing.

## Current Non-Goals
Do not create:

- deployment configuration,
- Vercel files,
- cloud backend,
- final demo script,
- presentation notes,
- submission checklist,
- production packaging.

## Final Response Expected From Claude
After Phase 05, Claude should stop and report only:

- the run command,
- completed local features,
- checks/tests run,
- commits made,
- known local limitations.
