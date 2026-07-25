from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation_dataset import (
    ArchiveImageStore,
    build_benchmark_manifest,
    validate_manifest,
)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (80, 100), color).save(output, format="JPEG")
    return output.getvalue()


def _write_fixture_datasets(root: Path) -> None:
    details = []
    references = []
    with ZipFile(root / "mst-e_data.zip", "w") as archive:
        for tone in range(1, 11):
            subject = f"subject_{tone}"
            golden = f"tone_{tone}_golden.jpg"
            adversarial = f"tone_{tone}_adversarial.jpg"
            references.append(
                {
                    "subject": subject,
                    "golden_image_id": golden,
                    "not_ideal_image_id": adversarial,
                }
            )
            for image_id, lighting, pose in (
                (golden, "well_lit", "facing_camera"),
                (adversarial, "poorly_lit", "side"),
            ):
                details.append(
                    {
                        "image_ID": image_id,
                        "pose": pose,
                        "lighting": lighting,
                        "mask": 0,
                        "subject_name": subject,
                        "MST": tone,
                    }
                )
                archive.writestr(
                    f"mst-e_data/{subject}/{image_id}",
                    _jpeg_bytes((120 + tone, 90, 70)),
                )
        archive.writestr(
            "mst-e_data/mst-e_image_details.csv",
            pd.DataFrame(details).to_csv(index=False),
        )
        archive.writestr(
            "mst-e_data/golden_and_adversarial_mst-e_image_ids.csv",
            pd.DataFrame(references).to_csv(index=False),
        )

    fairface_rows = []
    races = [
        "Black",
        "East Asian",
        "Indian",
        "Latino_Hispanic",
        "Middle Eastern",
        "Southeast Asian",
        "White",
    ]
    with ZipFile(root / "fairface-img-margin125-trainval.zip", "w") as archive:
        for index in range(1, 29):
            member = f"val/{index}.jpg"
            fairface_rows.append(
                {
                    "file": member,
                    "age": "20-29" if index % 2 else "40-49",
                    "gender": "Female" if index % 2 else "Male",
                    "race": races[(index - 1) % len(races)],
                    "service_test": False,
                }
            )
            archive.writestr(member, _jpeg_bytes((100, 100 + index, 80)))
    pd.DataFrame(fairface_rows).to_csv(
        root / "fairface_label_val.csv",
        index=False,
    )


def test_build_manifest_is_balanced_and_archive_valid(tmp_path):
    _write_fixture_datasets(tmp_path)

    manifest = build_benchmark_manifest(
        tmp_path,
        total_count=30,
        mste_count=20,
        seed=42,
    )

    assert len(manifest) == 30
    assert manifest["benchmark_id"].is_unique
    assert manifest["dataset"].value_counts().to_dict() == {
        "mste": 20,
        "fairface": 10,
    }
    mste = manifest[manifest["dataset"] == "mste"]
    assert mste["mst"].value_counts().sort_index().tolist() == [2] * 10
    assert set(mste["reference_role"]) == {"golden", "not_ideal"}
    assert not validate_manifest(manifest, tmp_path)


def test_archive_image_store_reads_without_extracting(tmp_path):
    _write_fixture_datasets(tmp_path)
    manifest = build_benchmark_manifest(
        tmp_path,
        total_count=30,
        mste_count=20,
    )

    with ArchiveImageStore(tmp_path) as store:
        image = store.load_rgb(manifest.iloc[0])

    assert isinstance(image, np.ndarray)
    assert image.shape == (100, 80, 3)
