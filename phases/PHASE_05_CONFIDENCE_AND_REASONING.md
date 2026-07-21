# Phase 05 — Confidence and Reasoning

## Goal
Add confidence scores and explanations for each recommendation.

## Tasks
1. Implement `src/confidence.py`.
2. Implement `src/explanation.py`.
3. Combine factors:
   - Delta E distance.
   - Region consistency.
   - Valid pixel ratio.
   - Face detection quality.
   - Top 1 vs Top 2 separation.
4. Add confidence percentage to each Top 3 shade.
5. Add explanation text to each result card.
6. Add warnings for difficult images.

## Suggested Confidence Formula
Start with an interpretable formula:

```text
base_score = exp(-delta_e / temperature)
relative_score = base_score / sum(base_scores_for_catalog_or_top_k)
quality_adjusted = relative_score * image_quality_score
```

Then clamp to a sensible range.

Alternative simple formula:

```text
confidence = weighted sum of:
- closeness score
- region consistency score
- valid pixel score
- face quality score
- top match separation score
```

## Explanation Template

```text
This shade is recommended because it has one of the lowest perceptual color differences from the extracted cheek/forehead/jawline skin tone. Confidence is adjusted based on lighting quality, region consistency, and how clearly this shade separates from nearby alternatives.
```

Make explanations specific when possible:
- Mention if undertone matches.
- Mention if confidence is reduced due to shadows.
- Mention if Top 1 and Top 2 are very close.

## Acceptance Criteria
- Every Top 3 shade has confidence.
- Every Top 3 shade has explanation.
- Bad lighting lowers confidence.
- Region disagreement lowers confidence.
- Explanations are deterministic and do not require an LLM API.


## Commit and Continue Rule
After this phase passes its acceptance criteria:

```bash
git status
git add .
git commit -m "phase-05: add confidence and explanations"
```

Then stop because Phase 05 is the final active phase for this local-app pass. Do not ask for permission to proceed. Do not push to GitHub.


## Final Stop Rule for Current Pass
This is the last active phase for the current local-app build.

After this phase passes:

```bash
git status
git add .
git commit -m "phase-05: add confidence reasoning and local completion"
```

Then stop. Do not continue to demo QA, testing-and-robustness phase, submission phase, deployment, Vercel, presentation, or packaging work.

Report only:
- how to run the local app,
- what local features are complete,
- what checks/tests were run,
- commits made,
- known local limitations.
