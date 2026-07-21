# 00 — Requirements Traceability

This document maps every assessment requirement to the implementation strategy.

## Objective
Build an AI solution that analyzes a facial image and recommends the most suitable cosmetic foundation shade from a predefined shade catalog.

## Current Delivery Target
Build and validate **ShadeSense AI** as a working local Streamlit app first. Deployment is not part of the current implementation scope.

## Requirement Mapping

| Assessment requirement | Planned implementation | Acceptance check |
|---|---|---|
| Accept a facial image | Streamlit `st.file_uploader` accepts JPG/PNG/JPEG. | User can upload an image locally and see it rendered. |
| Detect the face automatically | Use MediaPipe Face Landmarker / Face Mesh. | App detects one face and overlays landmarks or face mesh. |
| Identify cheeks, forehead, jawline | Use landmark-index polygon masks for cheeks, forehead, and jawline. | App displays separate masks for each region. |
| Handle different skin tones | Use median Lab color extraction and avoid fixed skin-tone thresholds as the main method. | Works on light, medium, tan, brown, and deep sample images. |
| Handle lighting and shadows | Apply mild white balance, luminance percentile filtering, and region quality scoring. | Shadow/highlight pixels are excluded from the extracted swatch. |
| Handle mild makeup reasonably | Exclude lips/eyes; use multiple regions; warn when region disagreement is high. | Heavy blush/lipstick does not dominate final color. |
| Use shade catalog | Load catalog from CSV and normalize shade colors into RGB/Lab. | Mock catalog loads; real catalog can replace it with minimal schema mapping. |
| Recommend best matching foundation shade | Compute perceptual color distance between extracted skin Lab and catalog shade Lab. | Lowest distance shade is ranked #1. |
| Return Top 3 shades | Sort all shades and return first three. | UI always shows Top 3 unless catalog has fewer than 3 shades. |
| Include confidence score | Combine distance score, region consistency, face quality, and top-match separation. | Each Top 3 item shows a confidence percentage. |
| Explain reasoning | Generate short deterministic explanation from distance, undertone/depth, and quality factors. | Each recommendation includes a readable reason. |
| Prepare live demo | Not part of the current automated build. The local app should still show visual stages so a demo can be prepared later. | Current scope stops at a working local app. |
| Explain approach, libraries, challenges, limitations | README + lightweight approach/limitations docs only. | No final demo script or submission packaging in this pass. |

## What Counts as “Flawless” for This Assessment

Flawless does not mean perfect cosmetic matching under every possible real-world photo. It means:

- The app runs without crashes.
- The pipeline is explainable.
- The output is visually inspectable.
- The confidence score reflects uncertainty.
- The implementation handles normal failure cases gracefully.
- The README clearly explains limitations and production improvements.

## Non-Goals

Do not spend time on these before the AI pipeline is stable:

- Full authentication.
- Database.
- E-commerce checkout.
- Heavy deployment work or Vercel implementation during the local-app phase.
- Training a foundation classifier from scratch.
- Polished landing page.
