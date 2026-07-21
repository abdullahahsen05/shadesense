# 05 — Autonomous Execution and Commit Plan

## Purpose
This project should be built by Claude Code from scratch with full local permissions. Claude should work continuously, but each phase must remain controlled and reviewable.

## App Name
Use **ShadeSense AI** as the app name in the Streamlit title and README.

## Current Scope — Local App Only
Only build the local working app.

Do not add or prepare:
- Vercel.
- Deployment.
- Cloud hosting.
- FastAPI.
- Authentication.
- Database.
- User accounts.
- Payment/SaaS features.
- Demo QA.
- Submission packaging.
- Presentation/demo script.

## Execution Rules for Claude
1. Start at Phase 00 and proceed sequentially through **Phase 05 only**.
2. Read the relevant phase file before writing code.
3. Do not skip phase acceptance criteria.
4. Run tests or smoke checks after each phase.
5. Make sensible Git commits after small stable additions.
6. Continue automatically to the next phase after a phase passes.
7. Stop after Phase 05 passes and report local run instructions.
8. Stop earlier only for real blockers, not for routine confirmation.
9. Do not push to GitHub; the developer will push manually.

## Active Phase List for This Pass

```text
PHASE_00_SETUP_AND_RULES.md
PHASE_01_FACE_DETECTION.md
PHASE_02_REGION_MASKS.md
PHASE_03_SKIN_EXTRACTION_AND_FILTERING.md
PHASE_04_CATALOG_AND_SHADE_MATCHING.md
PHASE_05_CONFIDENCE_AND_REASONING.md
```

Any demo/submission/Phase 06/Phase 07 work is intentionally excluded from this pass.

## Recommended Commit Cadence

Commit after each stable milestone:

```text
phase-00: initialize local Streamlit project
phase-01: add face detection and landmark overlay
phase-02: add cheek forehead jawline masks
phase-03: add robust skin extraction pipeline
phase-04: add shade catalog and top three matching
phase-05: add confidence reasoning and local completion
```

Additional small commits are allowed when useful:

```text
fix: handle no-face images gracefully
fix: correct RGB/BGR conversion in landmark pipeline
test: add catalog validation tests
refactor: separate Streamlit UI from matching logic
```

## Quality Gate Before Each Commit
Before every commit, Claude should run the smallest relevant check:

- Syntax check: `python -m compileall src app.py`
- Tests when available: `pytest`
- Local smoke check when UI changes: import-test the app and confirm `streamlit run app.py` is the run command
- Catalog check: load `data/shade_catalog_mock.csv` with pandas

## Phase 05 Completion Gate
Before stopping after Phase 05, Claude should verify the local app has:

- image upload,
- automatic face detection,
- visible cheek/forehead/jawline region masks,
- skin color extraction,
- mock catalog loading,
- Top 3 shade recommendations,
- confidence scores,
- short reasoning text,
- graceful no-face handling,
- local run instructions in README.

## Stop Conditions
Claude should stop only if:

- A dependency cannot be installed locally.
- MediaPipe/OpenCV fails due to platform issues that require user action.
- The real shade catalog arrives in an ambiguous format and requires mapping decisions.
- A proposed action would delete or rewrite major work outside the repo.
- A phase cannot pass after multiple focused fixes.

When stopped, Claude should write the reason and next action in `docs/blockers.md`.

## Strict Non-Goals Right Now
- No Vercel.
- No deployment.
- No FastAPI unless explicitly requested later.
- No database.
- No login/auth.
- No cloud file uploads.
- No paid APIs.
- No LLM dependency for core shade matching.
- No demo QA phase.
- No submission phase.
