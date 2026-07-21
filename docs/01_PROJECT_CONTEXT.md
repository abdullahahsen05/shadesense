# 01 — Project Context for Claude/Codex

## User Context
The developer is preparing a final-round AI Engineer hiring assessment. They passed screening and a technical interview. The employer explicitly allows coding tools, and the goal is to make the project robust, accurate, and locally working.

The developer is comfortable with Claude Code, Codex, Python, full-stack work, and Vercel. However, for this stage the project should be local-first so the AI pipeline can be completed before any deployment work.

## Business Problem
Cosmetic foundation matching is difficult because selfies vary by:

- Lighting temperature.
- Shadows and highlights.
- Camera white balance.
- Skin undertones.
- Surface redness.
- Mild makeup.
- Hair/beard occlusion.
- Face angle and expression.

The goal is not medical skin analysis. The goal is shade recommendation from a given product catalog.

## Core Technical Positioning
This should be presented as:

> An explainable computer vision and color-science pipeline for foundation shade recommendation.

Not:

> A generic AI chatbot or arbitrary image classifier.

## Why Not Train a CNN?
A supervised CNN needs labeled training examples mapping face photos to correct foundation shades. The assessment does not provide such a dataset, and the shade catalog is not yet available. A trained classifier would be hard to defend and likely inaccurate.

Instead, use:

- Pretrained face landmark detection.
- Deterministic region masks.
- Robust pixel filtering.
- Lab color space.
- Delta E / CIEDE2000 color difference.
- Confidence scoring.

## Expected Final Demo Story
Later, the final demo can say:

1. We do not average the whole face because lips, hair, shadows, eyes, and background distort skin color.
2. We detect facial landmarks and extract stable skin regions.
3. We filter unreliable pixels using luminance and saturation rules.
4. We estimate a representative skin color from multiple regions.
5. We convert both skin and shade catalog colors into Lab.
6. We rank catalog shades using perceptual color distance.
7. We return Top 3 recommendations because shade matching under uncontrolled lighting is uncertain.
8. We reduce confidence when image quality or region agreement is poor.

## App Name
Use this name consistently:

```text
ShadeSense AI
```

Why this name works:
- Short and easy to remember.
- Directly communicates shade recommendation.
- Professional enough for an assessment.
- Does not overpromise dermatology, medical skin analysis, or perfect cosmetic matching.

Do not overbrand it. This is an assessment, not a SaaS launch.


## Current Scope Lock
The current goal is a working local app only. Vercel/deployment can be discussed later after the local AI pipeline is accurate, robust, and locally working.
