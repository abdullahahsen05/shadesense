"""Skin-depth diagnostics for extraction sanity checks.

These helpers are diagnostic only. Shade ranking still uses CIEDE2000 as the
primary metric.
"""

from dataclasses import dataclass
import math


DEPTH_ORDER = {
    "fair": 0,
    "light": 1,
    "light-medium": 2,
    "medium": 3,
    "tan": 4,
    "deep": 5,
    "rich-deep": 6,
}


@dataclass
class SkinDepthDiagnostic:
    ita_degrees: float
    ita_category: str
    depth_category: str


def calculate_ita(l_value: float, b_value: float) -> float:
    """Return Individual Typology Angle in degrees from CIE Lab L* and b*."""
    denominator = b_value if abs(b_value) > 1e-6 else 1e-6
    return float(math.degrees(math.atan2(l_value - 50.0, denominator)))


def estimate_depth_from_lab_l(l_value: float) -> str:
    """Estimate broad shade depth from CIE Lab L*."""
    if l_value >= 85:
        return "fair"
    if l_value >= 75:
        return "light"
    if l_value >= 65:
        return "light-medium"
    if l_value >= 55:
        return "medium"
    if l_value >= 45:
        return "tan"
    if l_value >= 32:
        return "deep"
    return "rich-deep"


def estimate_depth_from_ita(ita_degrees: float) -> tuple[str, str]:
    """Return (ITA category label, broad shade-depth category)."""
    if ita_degrees > 55:
        return "very light", "fair"
    if ita_degrees > 41:
        return "light", "light"
    if ita_degrees > 28:
        return "intermediate", "medium"
    if ita_degrees > 10:
        return "tan", "tan"
    if ita_degrees > -30:
        return "brown", "deep"
    return "dark", "rich-deep"


def build_skin_depth_diagnostic(lab: tuple | list) -> SkinDepthDiagnostic:
    l_value = float(lab[0])
    b_value = float(lab[2])
    ita = calculate_ita(l_value, b_value)
    ita_category, ita_depth = estimate_depth_from_ita(ita)
    l_depth = estimate_depth_from_lab_l(l_value)
    # Keep L* as the stable broad shade-depth label, while preserving ITA as
    # diagnostic context. ITA can swing with b* and is not used for matching.
    depth_category = l_depth if abs(DEPTH_ORDER[l_depth] - DEPTH_ORDER[ita_depth]) <= 1 else l_depth
    return SkinDepthDiagnostic(
        ita_degrees=ita,
        ita_category=ita_category,
        depth_category=depth_category,
    )


def depth_distance(extracted_depth: str, shade_depth: str | None) -> int | None:
    if not shade_depth:
        return None
    shade = str(shade_depth)
    if extracted_depth not in DEPTH_ORDER or shade not in DEPTH_ORDER:
        return None
    return abs(DEPTH_ORDER[extracted_depth] - DEPTH_ORDER[shade])


def depth_match_status(extracted_depth: str, shade_depth: str | None) -> str:
    distance = depth_distance(extracted_depth, shade_depth)
    if distance is None:
        return "unknown"
    if distance == 0:
        return "aligned"
    if distance == 1:
        return "close"
    return "possible mismatch"


def depth_sanity_note(extracted_depth: str, shade_depth: str | None) -> str:
    status = depth_match_status(extracted_depth, shade_depth)
    if status == "aligned":
        return "Depth category aligns with the extracted skin-depth diagnostic."
    if status == "close":
        return "Depth category is close to the extracted skin-depth diagnostic."
    if status == "possible mismatch":
        return "Depth category differs slightly; color distance was still close."
    return "Catalog depth metadata is unavailable, so depth alignment is not assessed."
