# ShadeSense AI Validation Report

Use this template for manual validation. Do not fill results unless the case was
actually tested locally with an authorized image.

## Test Matrix

| Case ID | Skin Tone / Depth | Lighting | Pose / Framing | Makeup / Occlusion | Expected Behavior | Top 3 Output | Confidence Notes | Pass/Fail | Notes |
|---|---|---|---|---|---|---|---|---|---|
| V-001 | Fair | Even daylight | Front-facing | None | Detect face, balanced regions, plausible Top 3 |  |  |  |  |
| V-002 | Light | Indoor warm light | Front-facing | None | Detect possible color cast, keep recommendation usable |  |  |  |  |
| V-003 | Medium | Uneven side light | Front-facing | None | Warn about uneven lighting/shadow contrast |  |  |  |  |
| V-004 | Tan | Low light | Front-facing | None | Warn about underexposure, avoid hard failure |  |  |  |  |
| V-005 | Deep | Even daylight | Front-facing | None | Preserve valid deep skin pixels; avoid over-filtering |  |  |  |  |
| V-006 | Rich-deep | Low light | Front-facing | None | Adaptive filtering should keep skin signal with lower confidence |  |  |  |  |
| V-007 | Any | Overexposed / glare | Front-facing | None | Warn about overexposure/glare |  |  |  |  |
| V-008 | Any | Mixed color temperature | Front-facing | None | Warn about possible color cast |  |  |  |  |
| V-009 | Any | Even light | Slight head turn | None | Regions remain usable or warn gently |  |  |  |  |
| V-010 | Any | Even light | Front-facing | Bangs / hair near forehead | Forehead excluded or reduced if contaminated |  |  |  |  |
| V-011 | Any | Even light | Front-facing | Facial hair / jaw shadow | Jawline reduced, cheeks remain primary |  |  |  |  |
| V-012 | Any | Even light | Front-facing | Blush / strong cheek makeup | Patch/filter warnings; lower confidence if inconsistent |  |  |  |  |
| V-013 | Any | Even light | No face | None | Graceful no-face error |  |  |  |  |
| V-014 | Any | Even light | Multiple faces | None | Warn and select largest/most central face |  |  |  |  |

## Items To Record

- Selected catalog:
- Catalog shade count:
- Extracted RGB/Lab:
- Lighting quality score and warnings:
- Included regions:
- Not-used regions:
- Reduced-weight regions:
- Stable patches per region:
- Top 3 shades and Delta E:
- Confidence breakdown per shade:
- Close-match tie shown?:
- User-observed plausibility:
