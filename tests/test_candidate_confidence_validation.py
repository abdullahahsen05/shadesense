import pandas as pd

from scripts.validate_candidate_confidence import _markdown_table, _summary


def test_validation_summary_detects_candidate_differentiation_and_caps():
    rows = pd.DataFrame(
        [
            {
                "image": "one.jpg",
                "rank": 1,
                "color_fit": 0.90,
                "candidate_evidence": 0.88,
                "candidate_confidence": 0.66,
                "readiness_cap": 0.75,
                "catalog_evidence": 1.0,
                "stability_source": "shade_family",
            },
            {
                "image": "one.jpg",
                "rank": 2,
                "color_fit": 0.82,
                "candidate_evidence": 0.74,
                "candidate_confidence": 0.555,
                "readiness_cap": 0.75,
                "catalog_evidence": 1.0,
                "stability_source": "shade_family",
            },
        ]
    )
    captures = [
        {
            "image": "one.jpg",
            "success": True,
            "candidate_spread": 0.105,
        }
    ]

    result = _summary(rows, captures)

    assert result["images_with_at_least_1pp_candidate_spread"] == 1
    assert result["all_confidences_respect_readiness_cap"]
    assert result["catalog_evidence_is_constant"]


def test_markdown_table_has_no_optional_tabulate_dependency():
    rendered = _markdown_table(
        pd.DataFrame([{"shade": "A | B", "confidence": "72%"}])
    )

    assert "| shade | confidence |" in rendered
    assert r"A \| B" in rendered
