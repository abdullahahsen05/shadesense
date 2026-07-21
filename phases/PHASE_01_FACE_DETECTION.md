# Phase 01 — Face Detection and Landmarks

## Goal
Detect the face automatically and extract landmarks from uploaded images.

## Tasks
1. Implement `src/face_detection.py`.
2. Use MediaPipe Face Landmarker or Face Mesh.
3. Convert landmarks to pixel coordinates.
4. Return a structured `FaceDetectionResult`.
5. Add visualization for face landmarks in `src/visualization.py`.
6. Update Streamlit app to show landmark overlay.
7. Handle no-face and multiple-face cases gracefully.

## Implementation Notes
- Prefer one face for this assessment.
- If multiple faces are found, select the largest/most central face and show a warning.
- If no face is found, stop the pipeline and show a clear error.

## Acceptance Criteria
- Uploading a clear portrait shows landmarks.
- No-face image shows a friendly error.
- Multiple-face image shows a warning.
- App does not crash.

## Do Not Do Yet
- Do not perform shade matching.
- Do not build cheek masks yet unless landmarks are stable.


## Commit and Continue Rule
After this phase passes its acceptance criteria:

```bash
git status
git add .
git commit -m "phase-01: add face detection and landmark overlay"
```

Then continue automatically to the next phase. Do not ask for permission to proceed unless a true blocker appears. Do not push to GitHub.
