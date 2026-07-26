from pathlib import Path


def test_validation_report_template_exists_with_required_cases():
    report = Path("docs/validation_report.md")

    assert report.exists()
    text = report.read_text(encoding="utf-8").casefold()

    required_cases = [
        "fair",
        "light",
        "medium",
        "tan",
        "deep",
        "rich-deep",
        "yellow indoor lighting",
        "strong shadow",
        "side-angle",
        "sunglasses",
        "shadow",
        "no-face",
        "multi-face",
    ]
    required_columns = [
        "image filename",
        "skin-depth group",
        "lighting condition",
        "face detected",
        "regions used",
        "regions excluded/down-weighted",
        "extraction quality",
        "capture readiness",
        "candidate confidence",
        "top 3 shades",
        "visual mask quality",
        "result looks reasonable? yes/no",
        "notes",
    ]
    for required_text in required_cases + required_columns:
        assert required_text in text

    assert "do not fill results unless" in text


def test_demo_talking_points_contains_variation_handling_section():
    path = Path("docs/demo_talking_points.md")

    assert path.exists()
    text = path.read_text(encoding="utf-8").casefold()

    assert "how the system handles real-world variation" in text
    assert "why true skin color is hard from a single selfie" in text
    for required_text in [
        "different skin tones",
        "deep-skin-safe adaptive filtering",
        "shadows",
        "facial highlights",
        "mild makeup",
        "jawline and forehead contamination",
        "color correction safeguard",
        "capture readiness vs candidate confidence",
        "separated photo/extraction",
        "global readiness ceiling",
        "65% color",
        "25% shade-family stability",
        "10% catalog evidence",
        "public catalog limitations",
        "camera processing",
        "white balance",
    ]:
        assert required_text in text


def test_approach_explains_readiness_and_candidate_confidence_are_separate():
    path = Path("docs/approach.md")

    assert path.exists()
    text = path.read_text(encoding="utf-8").casefold()

    assert "skin extraction quality score" in text
    assert "capture readiness" in text
    assert "candidate confidence" in text
    assert "intentionally separate" in text
    assert "93%, 75%, and 55%" in text


def test_architecture_documents_candidate_confidence_formula_and_fallbacks():
    path = Path("docs/02_TECHNICAL_ARCHITECTURE.md")

    text = path.read_text(encoding="utf-8").casefold()

    assert "color_fit = exp(-distribution_aware_delta_e / 15)" in text
    assert "65% color_fit" in text
    assert "25% candidate_stability" in text
    assert "10% catalog_evidence" in text
    assert "exact_product_fallback" in text
    assert "missing factors are" in text
    assert "never silently treated as zero" in text
    assert "not calibrated" in text
