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
        "side",
        "shadow",
        "no face",
        "multiple faces",
    ]
    for required_case in required_cases:
        assert required_case in text

    assert "do not fill results unless" in text
    assert "pass/fail" in text
