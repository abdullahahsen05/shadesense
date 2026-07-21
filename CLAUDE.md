# CLAUDE.md — Foundation Shade AI Assessment

## Role
You are building a local-first AI/computer-vision assessment project for an AI Engineer final hiring round. The current goal is a **working local app only**. Do not prepare final submission materials yet.

## App Name
Use this app name throughout the local app and README:

```text
ShadeSense AI
```

Reason: it is short, professional, cosmetic-relevant, and does not overpromise medical or dermatology accuracy.

## Current Scope — Local Running App Only
Build a working local Streamlit app only.

Do not implement or prepare:
- Vercel or deployment.
- Cloud hosting.
- FastAPI/backend service.
- Authentication.
- Database.
- User accounts.
- SaaS/payment features.
- Final demo polish.
- Demo QA.
- Submission packaging.
- Presentation script.

Those can be discussed later after the local app is running correctly.

## Permissions and Autonomy
Claude Code has full local project access and permission to create, edit, refactor, install dependencies, run tests, run the Streamlit app locally, and make Git commits. Use that access responsibly.

Autonomous execution rule:
- Work phase by phase in order from **Phase 00 to Phase 05 only**.
- Do not execute any Phase 06/Phase 07/demo/submission work in this pass.
- Do not stop after every phase to ask for permission.
- After finishing a phase, run its acceptance checks, make a sensible Git commit, then continue to the next phase.
- After Phase 05 passes, stop and report:
  - how to run the local app,
  - what is implemented,
  - any known local limitations,
  - what commits were made.
- Stop earlier only for a true blocker: missing system dependency that cannot be installed, unclear real shade catalog format after it arrives, repeated failing checks that require human judgment, or a destructive action outside the project folder.
- Do not push to GitHub. The developer will push manually.
- Never edit files outside this repository unless explicitly necessary for the virtual environment or dependency installation.

## Assessment Objective
Build an AI solution that analyzes a facial image and recommends the most suitable cosmetic foundation shade from a predefined shade catalog.

The solution must:
- Accept a facial image.
- Detect the face automatically.
- Identify primary skin regions: cheeks, forehead, and jawline.
- Handle variations such as different skin tones, lighting, shadows, and mild makeup where reasonably possible.
- Use the provided shade catalog.
- Return Top 3 shade recommendations.
- Include a confidence score for each recommendation.
- Briefly explain the reasoning behind each recommendation.

## Implementation Philosophy
Prioritize a strong AI/CV pipeline over UI polish.

Do NOT build a black-box ML classifier for shade prediction. There is no labeled training dataset and the shade catalog may arrive late. Instead build an explainable pipeline:

```text
input image
→ face detection / landmarks
→ cheek, forehead, jawline masks
→ bad-pixel filtering
→ lighting correction / normalization
→ representative skin color extraction
→ Lab color conversion
→ perceptual color distance against catalog shades
→ Top 3 recommendations + confidence + explanation
```

## Local-First Stack
Use this local stack first:

```text
Python
Streamlit
OpenCV
MediaPipe
NumPy
Pandas
Pillow
scikit-image
pytest
```

Why local first:
- It lets us iterate quickly on the hard AI/CV parts.
- It avoids deployment issues while validating the local pipeline.
- It makes the app visual: original image, face landmarks, masks, extracted swatch, Top 3 shade results.

## Recommended Project Structure

```text
foundation-shade-ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── shade_catalog_mock.csv
│   └── sample_images/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── face_detection.py
│   ├── region_masks.py
│   ├── color_correction.py
│   ├── skin_extraction.py
│   ├── shade_catalog.py
│   ├── shade_matcher.py
│   ├── confidence.py
│   ├── explanation.py
│   └── visualization.py
│
├── tests/
│   ├── test_color_conversion.py
│   ├── test_shade_matcher.py
│   └── test_confidence.py
│
├── outputs/
│   └── debug/
│
└── docs/
    ├── approach.md
    └── limitations.md
```

## Hard Rules for Implementation
1. Keep modules small and testable.
2. Do not mix Streamlit UI logic with CV/matching logic.
3. Always show debug visualizations in the app.
4. Use median/trimmed statistics, not simple full-face averaging.
5. Never use lips, eyes, eyebrows, hair, beard, or background pixels as skin tone.
6. Use Lab color space for matching, not raw RGB distance.
7. Use Delta E / CIEDE2000 for final ranking if available.
8. Return Top 3, not just Top 1.
9. Confidence must reflect uncertainty; do not fake 99% confidence.
10. Add graceful failure messages when no face or multiple faces are detected.
11. Do not implement deployment in the current local-app phase.
12. Do not create final demo/submission materials in this current pass.

## Real Catalog Assumption
The real shade catalog has not arrived yet. Build with `data/shade_catalog_mock.csv` and make the catalog loader flexible.

Required normalized catalog schema:

```csv
shade_id,brand,shade_name,hex,r,g,b,undertone,depth,notes
```

The loader must support:
- HEX color values.
- RGB values.
- Optional undertone/depth fields.
- Later extension to swatch images.

## Local App Goal
The app should clearly show:
1. Uploaded image.
2. Detected face/landmarks.
3. Cheek, forehead, jawline masks.
4. Extracted skin pixels after filtering.
5. Final estimated skin swatch.
6. Top 3 shade cards.
7. Confidence scores.
8. Reasoning for each recommendation.
9. Warnings about poor lighting, blur, occlusion, makeup, or low confidence.

## Claude Code Execution Style
Work continuously but strictly phase by phase. Do not jump ahead, but also do not pause after each phase asking whether to continue.

For every phase:
1. Read the corresponding file in `/phases`.
2. Implement only the scope of that phase.
3. Run the phase acceptance checks.
4. Fix failures inside the current phase before moving on.
5. Make a small, sensible Git commit.
6. Continue to the next phase automatically.

Active phases for this pass:

```text
Phase 00 — Setup and Rules
Phase 01 — Face Detection and Landmarks
Phase 02 — Cheek, Forehead, and Jawline Masks
Phase 03 — Skin Extraction and Filtering
Phase 04 — Catalog Loading and Shade Matching
Phase 05 — Confidence and Reasoning
```

Do not create or execute Phase 06 or Phase 07 in this pass.

Recommended phase commit format:

```text
phase-00: initialize local Streamlit project
phase-01: add face detection and landmark overlay
phase-02: add facial region masks
phase-03: add skin extraction and filtering
phase-04: add catalog loading and shade matching
phase-05: add confidence reasoning and local completion
```

Commit rules:
- Commit after each passing phase.
- Also commit after any isolated feature/fix that is useful and stable.
- Use clear commit messages; avoid huge mixed commits.
- Do not commit broken code unless the commit is explicitly a checkpoint and labelled as such.
- Do not push to GitHub.

If a phase fails, fix the phase before moving forward. If blocked, document the blocker in `docs/blockers.md` and stop with a clear explanation.

## Final Stop Condition for This Pass
After Phase 05 passes, stop. Do not polish a final demo, do not create a presentation script, do not package a submission, and do not start deployment.

The final response should only include:
- local run command,
- implemented features,
- checks/tests run,
- Git commits made,
- known local limitations,
- suggested next phase for later discussion.

## Codex Usage
Use Codex mainly for:
- Reviewing generated code.
- Finding edge cases.
- Improving tests.
- Refactoring modules after they work.
- Checking numerical formulas.

Do not let Codex rewrite the whole project without preserving the architecture.
Do not ask Codex for deployment, final demo polish, or submission packaging right now.
