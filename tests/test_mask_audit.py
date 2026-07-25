import pandas as pd

from src.mask_audit import (
    REGIONS,
    build_mask_audit_manifest,
    summarize_mask_audit,
)


def _records():
    rows = []
    for dataset, count in (("mste", 80), ("fairface", 60)):
        for index in range(count):
            rows.append(
                {
                    "benchmark_id": f"{dataset}-{index}",
                    "dataset": dataset,
                    "archive_name": f"{dataset}.zip",
                    "archive_member": f"images/{index}.jpg",
                    "subject_id": f"s{index}",
                    "split": "development",
                    "mst": (index % 10) + 1 if dataset == "mste" else "",
                    "demographic_group": (
                        f"group-{index % 7}" if dataset == "fairface" else ""
                    ),
                    "lighting": "poorly_lit" if index % 3 == 0 else "well_lit",
                    "pose": "side" if index % 4 == 0 else "frontal",
                    "eyewear_detected": index % 5 == 0,
                    "pipeline_success": index % 17 != 0,
                    "extraction_quality_score": 40 + index % 50,
                    "capture_uncertainty_delta_e_p90": 2 + index % 12,
                    "pose_asymmetry": (index % 10) / 20,
                    "lighting_uneven": index % 3 == 0,
                }
            )
    return pd.DataFrame(rows)


def test_mask_audit_selects_requested_stratified_counts():
    audit = build_mask_audit_manifest(
        _records(),
        count=100,
        mste_count=60,
    )

    assert len(audit) == 100
    assert audit["benchmark_id"].is_unique
    assert audit["dataset"].value_counts().to_dict() == {
        "mste": 60,
        "fairface": 40,
    }
    assert not audit["reviewed"].any()
    for region in REGIONS:
        assert set(audit[f"{region}_review"]) == {"not_reviewed"}


def test_mask_audit_summary_uses_only_completed_reviews():
    audit = build_mask_audit_manifest(
        _records(),
        count=20,
        mste_count=10,
    )
    audit.loc[0, "reviewed"] = True
    audit.loc[0, "forehead_review"] = "major_contamination"
    for region in REGIONS[1:]:
        audit.loc[0, f"{region}_review"] = "clean"

    summary = summarize_mask_audit(audit)

    assert summary["reviewed_count"] == 1
    assert summary["review_completion"] == 0.05
    assert summary["regions"]["forehead"]["major_rate"] == 1.0
