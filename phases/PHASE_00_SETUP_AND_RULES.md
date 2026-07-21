# Phase 00 — Setup and Rules

## Goal
Create the project skeleton and install the local development stack.

## Tasks
1. Create the project structure exactly as described in `CLAUDE.md`.
2. Create a Python virtual environment.
3. Add `requirements.txt`.
4. Add `.gitignore`.
5. Create a minimal `README.md`.
6. Create `data/shade_catalog_mock.csv` with at least 15 mock shades across light, medium, tan, deep, warm, neutral, and cool tones.
7. Create empty modules under `src/`.
8. Add a smoke-test Streamlit app that only uploads and displays an image.

## Suggested `requirements.txt`

```text
streamlit
opencv-python
mediapipe
numpy
pandas
pillow
scikit-image
matplotlib
pytest
```

## Acceptance Criteria
- `python -m venv .venv` works.
- Dependencies install successfully.
- `streamlit run app.py` opens locally.
- User can upload and view an image.
- Mock shade catalog exists and loads manually with pandas.

## Do Not Do Yet
- Do not implement face landmarks.
- Do not implement matching.
- Do not add deployment.


## Commit and Continue Rule
After this phase passes its acceptance criteria:

```bash
git status
git add .
git commit -m "phase-00: initialize local Streamlit project"
```

Then continue automatically to the next phase. Do not ask for permission to proceed unless a true blocker appears. Do not push to GitHub.
