"""Shared constants and configuration for the ShadeSense AI pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEBUG_DIR = OUTPUTS_DIR / "debug"

SHADE_CATALOG_PATH = DATA_DIR / "shade_catalog_mock.csv"

APP_NAME = "ShadeSense AI"

MIN_FACE_SIZE_RATIO = 0.15
TOP_K_SHADES = 3
