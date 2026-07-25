"""Dataset adapters and deterministic benchmark-manifest construction."""

from __future__ import annotations

import hashlib
from io import BytesIO, TextIOWrapper
from pathlib import Path
import csv
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.image_io import open_rgb_image_with_metadata


MSTE_ARCHIVE = "mst-e_data.zip"
FAIRFACE_ARCHIVE = "fairface-img-margin125-trainval.zip"
FAIRFACE_LABELS = "fairface_label_val.csv"
MSTE_DETAILS_MEMBER = "mst-e_data/mst-e_image_details.csv"
MSTE_REFERENCE_MEMBER = "mst-e_data/golden_and_adversarial_mst-e_image_ids.csv"

# One identity per available MST level is kept out of development. Levels 7
# and 10 only have one identity in MST-E, so they are deliberately test-only.
MSTE_LOCKED_TEST_SUBJECTS = {
    "subject_18",  # MST 1
    "subject_13",  # MST 2
    "subject_15",  # MST 3
    "subject_9",   # MST 4
    "subject_6",   # MST 5
    "subject_3",   # MST 6
    "subject_5",   # MST 7 (singleton)
    "subject_17",  # MST 8
    "subject_4",   # MST 9
    "subject_12",  # MST 10 (singleton)
}

MANIFEST_COLUMNS = [
    "benchmark_id",
    "dataset",
    "archive_name",
    "archive_member",
    "image_id",
    "subject_id",
    "split",
    "mst",
    "demographic_group",
    "gender",
    "age",
    "lighting",
    "pose",
    "mask_present",
    "reference_role",
    "is_evaluation_reference",
    "expected_capture_label",
    "source_license",
]


def _read_zip_csv(archive: Path, member: str) -> pd.DataFrame:
    with ZipFile(archive) as zf:
        with zf.open(member) as raw:
            return pd.read_csv(TextIOWrapper(raw, encoding="utf-8-sig"))


def _stable_bucket(value: str, modulo: int = 10) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulo


def _fairface_split(image_id: str) -> str:
    bucket = _stable_bucket(image_id)
    if bucket < 6:
        return "development"
    if bucket < 8:
        return "validation"
    return "locked_test"


def _mste_capture_label(row: pd.Series) -> str:
    lighting = str(row["lighting"])
    pose = str(row["pose"])
    mask_present = int(row["mask"]) == 1
    if mask_present or (lighting == "poorly_lit" and pose in {"side", "bottom"}):
        return "recapture"
    if (
        lighting == "well_lit"
        and pose in {"facing_camera", "frontal"}
        and not mask_present
    ):
        return "usable"
    return "challenging"


def _balanced_round_robin(
    frame: pd.DataFrame,
    count: int,
    strata: list[str],
    seed: int,
) -> pd.DataFrame:
    """Select deterministic shuffled rows while spreading across strata."""
    if count >= len(frame):
        return frame.copy()
    rng = np.random.default_rng(seed)
    groups: list[list[int]] = []
    grouped = frame.groupby(strata, dropna=False, sort=True)
    for _, group in grouped:
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        groups.append(indices.tolist())
    rng.shuffle(groups)

    selected: list[int] = []
    while len(selected) < count and any(groups):
        next_groups = []
        for group in groups:
            if group and len(selected) < count:
                selected.append(group.pop())
            if group:
                next_groups.append(group)
        groups = next_groups
    return frame.loc[selected].copy()


def _mste_rows(dataset_root: Path, count: int, seed: int) -> pd.DataFrame:
    archive = dataset_root / MSTE_ARCHIVE
    details = _read_zip_csv(archive, MSTE_DETAILS_MEMBER)
    references = _read_zip_csv(archive, MSTE_REFERENCE_MEMBER)
    details = details[
        (details["pose"] != "video")
        & details["image_ID"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))
    ].copy()

    with ZipFile(archive) as zf:
        members = set(zf.namelist())
    details["archive_member"] = details.apply(
        lambda row: (
            f"mst-e_data/{row['subject_name']}/{row['image_ID']}"
        ),
        axis=1,
    )
    details = details[details["archive_member"].isin(members)].copy()

    golden = {
        (str(row["subject"]), str(row["golden_image_id"]))
        for _, row in references.iterrows()
    }
    adversarial = {
        (str(row["subject"]), str(row["not_ideal_image_id"]))
        for _, row in references.iterrows()
    }

    per_tone = [count // 10] * 10
    for index in range(count % 10):
        per_tone[index] += 1
    selected_parts = []
    for tone, tone_count in enumerate(per_tone, start=1):
        tone_rows = details[details["MST"].astype(int) == tone]
        reference_mask = tone_rows.apply(
            lambda row: (
                (str(row["subject_name"]), str(row["image_ID"])) in golden
                or (str(row["subject_name"]), str(row["image_ID"])) in adversarial
            ),
            axis=1,
        )
        required_references = tone_rows[reference_mask]
        if len(required_references) > tone_count:
            required_references = required_references.iloc[:tone_count]
        remaining = tone_rows.drop(required_references.index)
        sampled = _balanced_round_robin(
            remaining,
            tone_count - len(required_references),
            ["subject_name", "lighting", "pose", "mask"],
            seed + tone,
        )
        chosen = pd.concat([required_references, sampled])
        selected_parts.append(chosen)
    selected = pd.concat(selected_parts, ignore_index=True)

    def reference_role(row: pd.Series) -> str:
        key = (str(row["subject_name"]), str(row["image_ID"]))
        if key in golden:
            return "golden"
        if key in adversarial:
            return "not_ideal"
        return "ordinary"

    result = pd.DataFrame(
        {
            "dataset": "mste",
            "archive_name": MSTE_ARCHIVE,
            "archive_member": selected["archive_member"],
            "image_id": selected["image_ID"],
            "subject_id": selected["subject_name"],
            "split": selected["subject_name"].map(
                lambda value: (
                    "locked_test"
                    if value in MSTE_LOCKED_TEST_SUBJECTS
                    else "development"
                )
            ),
            "mst": selected["MST"].astype(int),
            "demographic_group": "",
            "gender": "",
            "age": "",
            "lighting": selected["lighting"],
            "pose": selected["pose"],
            "mask_present": selected["mask"].astype(int),
            "reference_role": selected.apply(reference_role, axis=1),
            "expected_capture_label": selected.apply(
                _mste_capture_label,
                axis=1,
            ),
            "source_license": (
                "TONL research/human-annotator-training only; no ML training"
            ),
        }
    )
    result["is_evaluation_reference"] = False
    for _, subject_rows in result.groupby("subject_id"):
        usable = subject_rows[
            subject_rows["expected_capture_label"] == "usable"
        ]
        candidates = usable if not usable.empty else subject_rows
        candidates = candidates.assign(
            _golden_first=(candidates["reference_role"] != "golden").astype(int)
        ).sort_values(["_golden_first", "image_id"])
        result.loc[candidates.index[0], "is_evaluation_reference"] = True
    return result


def _fairface_rows(dataset_root: Path, count: int, seed: int) -> pd.DataFrame:
    labels = pd.read_csv(dataset_root / FAIRFACE_LABELS)
    adult = labels[
        ~labels["age"].isin(["0-2", "3-9", "10-19"])
    ].copy()
    selected = _balanced_round_robin(
        adult,
        count,
        ["race", "gender", "age"],
        seed + 1000,
    )
    result = pd.DataFrame(
        {
            "dataset": "fairface",
            "archive_name": FAIRFACE_ARCHIVE,
            "archive_member": selected["file"],
            "image_id": selected["file"].map(lambda value: Path(value).name),
            "subject_id": selected["file"].map(
                lambda value: f"fairface_{Path(value).stem}"
            ),
            "split": selected["file"].map(_fairface_split),
            "mst": "",
            "demographic_group": selected["race"],
            "gender": selected["gender"],
            "age": selected["age"],
            "lighting": "unlabelled",
            "pose": "unlabelled",
            "mask_present": "",
            "reference_role": "ordinary",
            "is_evaluation_reference": False,
            "expected_capture_label": "review_required",
            "source_license": "CC BY 4.0",
        }
    )
    return result


def build_benchmark_manifest(
    dataset_root: str | Path,
    *,
    total_count: int = 400,
    mste_count: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Create the frozen, balanced benchmark manifest."""
    dataset_root = Path(dataset_root)
    fairface_count = total_count - mste_count
    if total_count <= 0 or mste_count <= 0 or fairface_count <= 0:
        raise ValueError("Benchmark counts must include both MST-E and FairFace.")
    required = [MSTE_ARCHIVE, FAIRFACE_ARCHIVE, FAIRFACE_LABELS]
    missing = [name for name in required if not (dataset_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset folder is missing required files: " + ", ".join(missing)
        )

    manifest = pd.concat(
        [
            _mste_rows(dataset_root, mste_count, seed),
            _fairface_rows(dataset_root, fairface_count, seed),
        ],
        ignore_index=True,
    )
    manifest.insert(
        0,
        "benchmark_id",
        [
            f"{dataset}-{index:04d}"
            for index, dataset in enumerate(manifest["dataset"], start=1)
        ],
    )
    manifest = manifest[MANIFEST_COLUMNS]
    if len(manifest) != total_count:
        raise ValueError(
            f"Expected {total_count} benchmark rows, produced {len(manifest)}."
        )
    if manifest["benchmark_id"].duplicated().any():
        raise ValueError("Benchmark IDs must be unique.")
    return manifest


def validate_manifest(
    manifest: pd.DataFrame,
    dataset_root: str | Path,
) -> list[str]:
    """Return structural validation errors for a benchmark manifest."""
    errors = []
    missing_columns = [
        column for column in MANIFEST_COLUMNS if column not in manifest.columns
    ]
    if missing_columns:
        return [f"Missing manifest columns: {missing_columns}"]
    if manifest["benchmark_id"].duplicated().any():
        errors.append("Duplicate benchmark IDs were found.")

    dataset_root = Path(dataset_root)
    by_archive = manifest.groupby("archive_name")
    for archive_name, rows in by_archive:
        archive_path = dataset_root / archive_name
        if not archive_path.exists():
            errors.append(f"Missing archive: {archive_path}")
            continue
        with ZipFile(archive_path) as zf:
            members = set(zf.namelist())
        missing_members = [
            member
            for member in rows["archive_member"].astype(str)
            if member not in members
        ]
        if missing_members:
            errors.append(
                f"{archive_name} is missing {len(missing_members)} selected member(s)."
            )
    return errors


class ArchiveImageStore:
    """Reusable ZIP handles for loading selected benchmark images."""

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self._archives: dict[str, ZipFile] = {}

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def load_rgb(self, row: pd.Series) -> np.ndarray:
        image, _ = self.load_rgb_with_metadata(row)
        return image

    def load_rgb_with_metadata(
        self,
        row: pd.Series,
    ) -> tuple[np.ndarray, dict]:
        archive_name = str(row["archive_name"])
        archive = self._archives.get(archive_name)
        if archive is None:
            archive = ZipFile(self.dataset_root / archive_name)
            self._archives[archive_name] = archive
        data = archive.read(str(row["archive_member"]))
        image, metadata = open_rgb_image_with_metadata(BytesIO(data))
        return np.asarray(image), metadata.as_dict()
