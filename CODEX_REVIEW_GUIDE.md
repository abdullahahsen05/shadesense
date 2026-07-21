# CODEX_REVIEW_GUIDE.md

Use Codex as a reviewer and second engineer after each active local-app phase.

## Review Scope
The current app is **ShadeSense AI**, a local-first Streamlit project. Codex should review correctness, robustness, tests, and explainability for a working local app only. Codex should not suggest deployment, Vercel, database, auth, SaaS features, demo QA, or submission packaging during the current build.

## Git Rule
After Codex-suggested fixes are applied and checks pass, make a focused Git commit. Do not push to GitHub.

## Active Reviews for This Pass

### After Phase 00
Prompt:
```text
Review this Python/Streamlit project skeleton for a local AI foundation shade matching app. Check if the structure is clean, dependencies are reasonable, and the app can be extended phase by phase. Do not rewrite everything; only suggest issues and focused fixes. Do not suggest deployment or demo/submission work.
```

### After Phase 01
Prompt:
```text
Review the MediaPipe face detection and landmark extraction code. Check for failure cases: no face, multiple faces, image color format mistakes, landmark coordinate conversion, and Streamlit display issues. Suggest minimal fixes only. Do not suggest deployment or demo/submission work.
```

### After Phase 02
Prompt:
```text
Review the region mask code for cheeks, forehead, and jawline. Check whether masks are likely to include lips, eyes, hairline, or background. Suggest safer landmark polygons and validation checks. Do not suggest deployment or demo/submission work.
```

### After Phase 03
Prompt:
```text
Review the skin extraction and filtering pipeline. Check if luminance filtering, saturation filtering, median color extraction, and region consistency logic are robust. Look for color-space conversion bugs and over-filtering. Do not suggest deployment or demo/submission work.
```

### After Phase 04
Prompt:
```text
Review catalog loading and shade matching. Check HEX/RGB parsing, Lab conversion, Delta E computation, sorting, Top 3 output, and invalid catalog handling. Verify that raw RGB distance is not used as the final ranking. Do not suggest deployment or demo/submission work.
```

### After Phase 05
Prompt:
```text
Review confidence scoring, explanation generation, and the local Streamlit app flow. Check whether the app fulfills the local requirements: upload image, detect face, show cheeks/forehead/jawline regions, extract skin tone, load mock catalog, return Top 3 shades, show confidence, explain reasoning, and handle no-face inputs gracefully. Do not suggest deployment, demo QA, presentation script, or submission packaging.
```

## Final Local-App Review Only
Prompt:
```text
Act as a strict code reviewer for a local AI/CV app. Review whether ShadeSense AI runs locally and satisfies the core assessment functionality: face image input, face detection, cheeks/forehead/jawline analysis, robustness to lighting/skin tones/mild makeup where reasonably possible, catalog matching, Top 3 shades, confidence scores, and reasoning. Identify only issues that affect the local app's correctness or reliability. Do not suggest deployment or submission polish.
```
