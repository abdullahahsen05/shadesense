"""Representative skin color extraction and filtering from masked face regions."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

from src.depth_diagnostics import build_skin_depth_diagnostic

REGION_NAMES = ["forehead", "left_cheek", "right_cheek", "jawline"]
CHEEK_NAMES = ("left_cheek", "right_cheek")
FOREHEAD_NAME = "forehead"
JAWLINE_NAME = "jawline"

LUMINANCE_LOWER_PERCENTILE = 10
LUMINANCE_UPPER_PERCENTILE = 90
SATURATION_UPPER_PERCENTILE = 92
MAX_ABSOLUTE_SATURATION = 170.0
MIN_VALID_PIXELS_PER_REGION = 100
REGION_DISAGREEMENT_THRESHOLD = 12.0  # Lab distance considered "high disagreement"
CHEEK_AGREEMENT_LAB_DISTANCE = 8.0
OPTIONAL_VS_CHEEK_DOWNWEIGHT_LAB_DISTANCE = 14.0
OPTIONAL_REGION_BASE_WEIGHT = 0.55
PATCH_SIZE = 18
PATCH_STRIDE = 12
MIN_ADAPTIVE_PATCH_SIZE = 12
MAX_ADAPTIVE_PATCH_SIZE = 36
BOOTSTRAP_ITERATIONS = 96
BOOTSTRAP_SEED = 20260724
MIN_VALID_PIXELS_PER_PATCH = 45
MIN_STABLE_PATCHES_PER_REGION = 2
MAX_STABLE_PATCHES_PER_REGION = 8
MAX_PATCH_LAB_STD = 8.0
MAX_PATCH_RGB_STD = 34.0
PATCH_LUMINANCE_MARGIN = 4.0
PATCH_HIGHLIGHT_L_MARGIN = 3.0
PATCH_SHADOW_L_MARGIN = 3.0
PATCH_STRONG_CONTRAST_L = 18.0
PATCH_HIGHLIGHT_PIXEL_RATIO = 0.18
PATCH_WASHED_OUT_CHROMA = 9.0
PATCH_LOW_SATURATION = 35.0
MAKEUP_INFLUENCE_RATIO = 0.18
HIGHLIGHT_INFLUENCE_RATIO = 0.18
MAKEUP_DOWNWEIGHT_FACTOR = 0.75
FOREHEAD_HIGHLIGHT_DOWNWEIGHT_FACTOR = 0.18
LOW_RELIABILITY_DOWNWEIGHT_FACTOR = 0.7
REGION_RELIABILITY_THRESHOLD = 0.45
JAWLINE_QUALITY_DELTA_E_THRESHOLD = 10.0
JAWLINE_HIGH_LAB_STD = 7.0
JAWLINE_LOW_VALID_RATIO = 0.45

# Forehead is useful but optional: if it disagrees strongly with the cheek
# tone (full Lab distance), it is excluded outright — likely hair/fringe or
# shadow contamination rather than skin.
FOREHEAD_VS_CHEEK_OUTLIER_LAB_DISTANCE = 20.0

# Jawline is never excluded outright, but facial hair or chin/neck shadow
# tends to make it specifically darker (not just differently colored) than
# the cheeks, so a lightness (L*) gap beyond this threshold reduces —
# rather than removes — its weight in the final combination.
JAWLINE_DOWNWEIGHT_FACTOR = 0.35
CHEEK_AREA_IMBALANCE_WARNING_RATIO = 0.45
CHEEK_FORESHORTENED_AREA_RATIO = 0.55
CHEEK_FORESHORTENED_WEIGHT = 0.68
REGION_CONTRIBUTION_CAPS = {
    "left_cheek": 0.45,
    "right_cheek": 0.45,
    "forehead": 0.15,
    "jawline": 0.30,
}


@dataclass(frozen=True)
class SkinPatchEvidence:
    region: str
    rgb: tuple
    lab: tuple
    quality: float
    is_midtone: bool
    lab_std: float
    rgb_std: float
    luminance_median: float
    luminance_contrast: float
    highlight_ratio: float
    shadow_ratio: float
    top: int
    left: int
    size: int


@dataclass
class RegionSkinResult:
    name: str
    total_pixel_count: int
    valid_pixel_count: int
    valid_ratio: float
    median_rgb: tuple | None
    median_lab: tuple | None
    reliable: bool  # had enough valid pixels after filtering
    excluded: bool = False  # excluded entirely from the final combination
    exclusion_reason: str | None = None
    weight_multiplier: float = 1.0  # down-weighting applied within combination (1.0 = full weight)
    downweight_reason: str | None = None
    stable_patch_count: int = 0
    stable_patch_rgbs: list = field(default_factory=list)
    stable_patch_labs: list = field(default_factory=list)
    stable_patch_quality_scores: list = field(default_factory=list)
    stable_patch_midtone_flags: list = field(default_factory=list)
    patch_fallback_used: bool = False
    lab_std: float = 0.0
    rgb_std: float = 0.0
    shadow_highlight_ratio: float = 0.0
    reliability_score: float = 0.0
    makeup_influence_detected: bool = False
    specular_highlight_detected: bool = False
    highlight_patches_rejected: int = 0
    shadow_patches_rejected: int = 0
    midtone_patch_count: int = 0
    quality_score: float = 0.0
    quality_label: str = "excluded"
    role: str = "excluded"
    quality_reasons: list = field(default_factory=list)
    quality_warnings: list = field(default_factory=list)
    patch_evidence: list[SkinPatchEvidence] = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def status_label(self) -> str:
        """Human-readable status. A region can only ever be labeled
        "reliable"/"included" XOR "excluded" — never both, so the UI can't
        show contradictory statuses for the same region."""
        if not self.reliable:
            return "insufficient pixels"
        if self.excluded:
            return "excluded"
        if self.weight_multiplier < 1.0:
            return "included (reduced weight)"
        return "included"

    @property
    def status_reason(self) -> str | None:
        if self.excluded:
            return self.exclusion_reason
        if self.weight_multiplier < 1.0:
            return self.downweight_reason
        return None


@dataclass
class SkinToneResult:
    rgb: tuple
    lab: tuple
    region_results: dict
    quality_score: float
    region_consistency: float = 0.0
    avg_valid_pixel_ratio: float = 0.0
    cheek_area_balance: float = 1.0
    usable_region_count: int = 0
    included_region_names: list = field(default_factory=list)
    excluded_region_names: list = field(default_factory=list)
    extraction_quality_reasons: list = field(default_factory=list)
    patch_voting_diagnostics: dict = field(default_factory=dict)
    stability_diagnostics: dict = field(default_factory=dict)
    foundation_target_rgb: tuple | None = None
    foundation_target_lab: tuple | None = None
    foundation_target_active: bool = False
    foundation_target_reason: str = ""
    foundation_target_diagnostics: dict = field(default_factory=dict)
    bootstrap_labs: list = field(default_factory=list)
    uncertainty_diagnostics: dict = field(default_factory=dict)
    depth_estimate: str | None = None
    ita_degrees: float | None = None
    ita_category: str | None = None
    warnings: list = field(default_factory=list)
    success: bool = True


def _to_lab(pixels_rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) uint8 RGB pixel array to (N, 3) Lab floats."""
    lab = rgb2lab(pixels_rgb.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)
    _assert_skimage_lab_scale(lab)
    return lab


def _assert_skimage_lab_scale(lab: np.ndarray) -> None:
    """Guard against accidental OpenCV Lab scale entering the pipeline."""
    if lab.size == 0:
        return
    l_values = np.asarray(lab, dtype=np.float64).reshape(-1, 3)[:, 0]
    if np.nanmin(l_values) < -1e-6 or np.nanmax(l_values) > 100.0 + 1e-6:
        raise ValueError("Lab values must use skimage/CIE L*=0-100 scale, not OpenCV L=0-255.")


def _to_saturation(pixels_rgb: np.ndarray) -> np.ndarray:
    """Return the HSV saturation channel (0-255) for an (N, 3) uint8 RGB pixel array."""
    hsv = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv.reshape(-1, 3)[:, 1].astype(np.float64)


def _filter_skin_pixels(pixels_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively filter masked pixels using only region-relative extremes."""
    if len(pixels_rgb) == 0:
        return pixels_rgb, np.empty((0, 3), dtype=np.float64)

    lab = _to_lab(pixels_rgb)
    saturation = _to_saturation(pixels_rgb)
    luminance = lab[:, 0]

    lum_low = float(np.percentile(luminance, LUMINANCE_LOWER_PERCENTILE))
    lum_high = float(np.percentile(luminance, LUMINANCE_UPPER_PERCENTILE))
    if lum_low >= lum_high:
        lum_low, lum_high = float(np.min(luminance)), float(np.max(luminance))

    sat_high = min(
        float(np.percentile(saturation, SATURATION_UPPER_PERCENTILE)),
        MAX_ABSOLUTE_SATURATION,
    )

    red_dominance = (
        (pixels_rgb[:, 0].astype(np.int16) - pixels_rgb[:, 1].astype(np.int16) > 70)
        & (pixels_rgb[:, 0].astype(np.int16) - pixels_rgb[:, 2].astype(np.int16) > 85)
        & (saturation > 130)
    )

    keep = (
        (luminance >= lum_low)
        & (luminance <= lum_high)
        & (saturation <= sat_high)
        & (~red_dominance)
    )
    return pixels_rgb[keep], lab[keep]


def _stable_patch_medians(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    region_name: str = "unknown",
    region_luminance_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Return RGB/Lab medians from the most stable patches in a mask."""
    ys, xs = np.where(mask > 0)
    stats = {
        "highlight_patches_rejected": 0,
        "shadow_patches_rejected": 0,
        "midtone_patch_count": 0,
        "selected_patch_quality_scores": [],
        "selected_midtone_flags": [],
        "patch_evidence": [],
        "patch_size": 0,
        "patch_stride": 0,
    }
    if len(xs) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), 0, stats

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch_size = int(
        np.clip(
            round(np.sqrt(len(xs)) * 0.22),
            MIN_ADAPTIVE_PATCH_SIZE,
            MAX_ADAPTIVE_PATCH_SIZE,
        )
    )
    patch_stride = max(8, int(round(patch_size * 2.0 / 3.0)))
    stats["patch_size"] = patch_size
    stats["patch_stride"] = patch_stride
    stable = []

    for top in range(y0, max(y0 + 1, y1 - patch_size + 1), patch_stride):
        for left in range(x0, max(x0 + 1, x1 - patch_size + 1), patch_stride):
            patch_mask = mask[top : top + patch_size, left : left + patch_size]
            patch_pixels = image_rgb[top : top + patch_size, left : left + patch_size][patch_mask > 0]
            if len(patch_pixels) < MIN_VALID_PIXELS_PER_PATCH:
                continue

            patch_raw_lab = _to_lab(patch_pixels)
            patch_raw_l = patch_raw_lab[:, 0]
            patch_l_median = float(np.median(patch_raw_l))
            patch_l_contrast = float(np.percentile(patch_raw_l, 95) - np.percentile(patch_raw_l, 5))
            patch_highlight_ratio = float(np.mean(patch_raw_l > 92.0))
            patch_shadow_ratio = float(np.mean(patch_raw_l < 12.0))
            patch_chroma = float(np.linalg.norm(np.median(patch_raw_lab, axis=0)[1:3]))
            patch_saturation = float(np.median(_to_saturation(patch_pixels)))

            valid_rgb, valid_lab = _filter_skin_pixels(patch_pixels)
            if len(valid_rgb) < MIN_VALID_PIXELS_PER_PATCH:
                continue

            lab_std = float(np.mean(np.std(valid_lab, axis=0)))
            rgb_std = float(np.mean(np.std(valid_rgb.astype(np.float64), axis=0)))
            if lab_std > MAX_PATCH_LAB_STD or rgb_std > MAX_PATCH_RGB_STD:
                continue
            luminance_mid_penalty = 0.0
            is_midtone = False
            if region_luminance_bounds is not None:
                if len(region_luminance_bounds) >= 4:
                    low_l, high_l, mid_low_l, mid_high_l = region_luminance_bounds[:4]
                else:
                    low_l, high_l = region_luminance_bounds[:2]
                    mid_low_l, mid_high_l = low_l, high_l
                bright_pixel_ratio = float(np.mean(patch_raw_l > mid_high_l + PATCH_HIGHLIGHT_L_MARGIN))
                if patch_l_median < low_l - PATCH_SHADOW_L_MARGIN:
                    stats["shadow_patches_rejected"] += 1
                    continue
                if patch_l_median > high_l + PATCH_HIGHLIGHT_L_MARGIN:
                    stats["highlight_patches_rejected"] += 1
                    continue
                washed_out_highlight = (
                    patch_l_median >= high_l - PATCH_HIGHLIGHT_L_MARGIN
                    and (patch_chroma < PATCH_WASHED_OUT_CHROMA or patch_saturation < PATCH_LOW_SATURATION)
                )
                contrast_highlight = (
                    patch_l_contrast > PATCH_STRONG_CONTRAST_L
                    and bright_pixel_ratio >= PATCH_HIGHLIGHT_PIXEL_RATIO
                )
                if washed_out_highlight or contrast_highlight:
                    stats["highlight_patches_rejected"] += 1
                    continue
                if mid_low_l <= patch_l_median <= mid_high_l:
                    stats["midtone_patch_count"] += 1
                    is_midtone = True
                else:
                    mid_center = (mid_low_l + mid_high_l) / 2.0
                    mid_width = max(mid_high_l - mid_low_l, 1.0)
                    luminance_mid_penalty = abs(patch_l_median - mid_center) / mid_width

            patch_quality = float(
                np.clip(
                    1.0
                    - 0.45 * min(lab_std / max(MAX_PATCH_LAB_STD, 1.0), 1.0)
                    - 0.25 * min(rgb_std / max(MAX_PATCH_RGB_STD, 1.0), 1.0)
                    - 0.20 * min(luminance_mid_penalty, 1.0)
                    + (0.10 if is_midtone else 0.0),
                    0.05,
                    1.0,
                )
            )
            stable.append(
                (
                    lab_std + rgb_std / 10.0 + luminance_mid_penalty,
                    np.median(valid_rgb, axis=0),
                    np.median(valid_lab, axis=0),
                    patch_quality,
                    is_midtone,
                    patch_l_median,
                    patch_l_contrast,
                    patch_highlight_ratio,
                    patch_shadow_ratio,
                    lab_std,
                    rgb_std,
                    top,
                    left,
                )
            )

    if len(stable) < MIN_STABLE_PATCHES_PER_REGION:
        return np.empty((0, 3)), np.empty((0, 3)), len(stable), stats

    stable.sort(key=lambda item: item[0])
    # Keep high-quality evidence while sampling across the candidate sequence
    # so an equally stable shadowed half cannot monopolize all retained
    # patches merely because it was scanned first.
    pool = stable[: min(len(stable), MAX_STABLE_PATCHES_PER_REGION * 3)]
    selected_count = min(len(pool), MAX_STABLE_PATCHES_PER_REGION)
    selected_indices = np.linspace(
        0,
        len(pool) - 1,
        num=selected_count,
        dtype=int,
    )
    selected = [pool[int(idx)] for idx in dict.fromkeys(selected_indices.tolist())]
    rgb_medians = np.stack([item[1] for item in selected])
    lab_medians = np.stack([item[2] for item in selected])
    stats["selected_patch_quality_scores"] = [item[3] for item in selected]
    stats["selected_midtone_flags"] = [item[4] for item in selected]
    stats["patch_evidence"] = [
        SkinPatchEvidence(
            region=region_name,
            rgb=tuple(np.round(item[1]).astype(int).tolist()),
            lab=tuple(np.asarray(item[2], dtype=np.float64).tolist()),
            quality=float(item[3]),
            is_midtone=bool(item[4]),
            luminance_median=float(item[5]),
            luminance_contrast=float(item[6]),
            highlight_ratio=float(item[7]),
            shadow_ratio=float(item[8]),
            lab_std=float(item[9]),
            rgb_std=float(item[10]),
            top=int(item[11]),
            left=int(item[12]),
            size=patch_size,
        )
        for item in selected
    ]
    return rgb_medians, lab_medians, len(selected), stats


def _extract_region(image_rgb: np.ndarray, mask: np.ndarray, name: str) -> RegionSkinResult:
    warnings: list = []
    pixels = image_rgb[mask > 0]
    total_count = int(len(pixels))

    if total_count == 0:
        return RegionSkinResult(
            name=name,
            total_pixel_count=0,
            valid_pixel_count=0,
            valid_ratio=0.0,
            median_rgb=None,
            median_lab=None,
            reliable=False,
            warnings=[f"No pixels found for {name}; region mask was empty."],
        )

    raw_lab = _to_lab(pixels)
    raw_luminance = raw_lab[:, 0]
    raw_saturation = _to_saturation(pixels)
    region_luminance_bounds = (
        float(np.percentile(raw_luminance, LUMINANCE_LOWER_PERCENTILE)),
        float(np.percentile(raw_luminance, LUMINANCE_UPPER_PERCENTILE)),
        float(np.percentile(raw_luminance, 25)),
        float(np.percentile(raw_luminance, 75)),
    )
    valid_pixels_rgb, valid_lab = _filter_skin_pixels(pixels)
    valid_count = int(len(valid_pixels_rgb))
    valid_ratio = valid_count / total_count if total_count else 0.0
    shadow_highlight_ratio = float(1.0 - valid_ratio)
    red_pink = (
        (pixels[:, 0].astype(np.int16) - pixels[:, 1].astype(np.int16) > 70)
        & (pixels[:, 0].astype(np.int16) - pixels[:, 2].astype(np.int16) > 85)
        & (raw_saturation > 130)
    )
    median_luminance = float(np.median(raw_luminance))
    high_luminance = float(np.percentile(raw_luminance, 96))
    bright_highlight = (raw_luminance >= high_luminance) & (
        raw_luminance - median_luminance > PATCH_LUMINANCE_MARGIN * 2
    )
    makeup_ratio = float(np.mean(red_pink)) if total_count else 0.0
    highlight_ratio = float(np.mean(bright_highlight & (raw_saturation < 80))) if total_count else 0.0
    makeup_influence_detected = (
        name in CHEEK_NAMES and makeup_ratio >= MAKEUP_INFLUENCE_RATIO
    ) or highlight_ratio >= HIGHLIGHT_INFLUENCE_RATIO

    if valid_count < MIN_VALID_PIXELS_PER_REGION:
        warnings.append(
            f"{name.replace('_', ' ').title()} has only {valid_count} valid pixels "
            f"after filtering (minimum recommended: {MIN_VALID_PIXELS_PER_REGION})."
        )

    if valid_count == 0:
        return RegionSkinResult(
            name=name,
            total_pixel_count=total_count,
            valid_pixel_count=0,
            valid_ratio=0.0,
            median_rgb=None,
            median_lab=None,
            reliable=False,
            warnings=warnings,
        )

    patch_rgb, patch_lab, stable_patch_count, patch_stats = _stable_patch_medians(
        image_rgb,
        mask,
        region_name=name,
        region_luminance_bounds=region_luminance_bounds,
    )
    patch_fallback_used = len(patch_rgb) == 0
    if patch_fallback_used:
        median_rgb = tuple(np.median(valid_pixels_rgb, axis=0).astype(int).tolist())
        median_lab = tuple(np.median(valid_lab, axis=0).tolist())
        lab_std = float(np.mean(np.std(valid_lab, axis=0)))
        rgb_std = float(np.mean(np.std(valid_pixels_rgb.astype(np.float64), axis=0)))
        if stable_patch_count > 0:
            warnings.append(
                f"{name.replace('_', ' ').title()} had only {stable_patch_count} stable patch(es); "
                "using full-region median fallback."
            )
    else:
        median_rgb = tuple(np.median(patch_rgb, axis=0).astype(int).tolist())
        median_lab = tuple(np.median(patch_lab, axis=0).tolist())
        lab_std = float(np.mean(np.std(patch_lab, axis=0)))
        rgb_std = float(np.mean(np.std(patch_rgb.astype(np.float64), axis=0)))

    stable_patch_score = min(stable_patch_count / max(MIN_STABLE_PATCHES_PER_REGION, 1), 1.0)
    variance_score = float(np.clip(1.0 - lab_std / max(MAX_PATCH_LAB_STD, 1.0), 0.0, 1.0))
    reliability_score = float(
        np.clip(
            0.45 * valid_ratio
            + 0.25 * stable_patch_score
            + 0.20 * variance_score
            + 0.10 * (1.0 - min(shadow_highlight_ratio, 1.0)),
            0.0,
            1.0,
        )
    )
    specular_highlight_detected = patch_stats["highlight_patches_rejected"] > 0 or highlight_ratio >= HIGHLIGHT_INFLUENCE_RATIO
    if makeup_influence_detected:
        warnings.append(f"{name.replace('_', ' ').title()}: possible makeup/highlight influence detected.")
    if specular_highlight_detected:
        warnings.append(f"{name.replace('_', ' ').title()}: possible specular highlight influence detected.")

    return RegionSkinResult(
        name=name,
        total_pixel_count=total_count,
        valid_pixel_count=valid_count,
        valid_ratio=valid_ratio,
        median_rgb=median_rgb,
        median_lab=median_lab,
        reliable=valid_count >= MIN_VALID_PIXELS_PER_REGION,
        stable_patch_count=stable_patch_count if not patch_fallback_used else 0,
        stable_patch_rgbs=[tuple(np.round(rgb).astype(int).tolist()) for rgb in patch_rgb] if not patch_fallback_used else [],
        stable_patch_labs=[tuple(lab.astype(float).tolist()) for lab in patch_lab] if not patch_fallback_used else [],
        stable_patch_quality_scores=patch_stats["selected_patch_quality_scores"] if not patch_fallback_used else [],
        stable_patch_midtone_flags=patch_stats["selected_midtone_flags"] if not patch_fallback_used else [],
        patch_fallback_used=patch_fallback_used,
        lab_std=lab_std,
        rgb_std=rgb_std,
        shadow_highlight_ratio=shadow_highlight_ratio,
        reliability_score=reliability_score,
        makeup_influence_detected=makeup_influence_detected,
        specular_highlight_detected=specular_highlight_detected,
        highlight_patches_rejected=patch_stats["highlight_patches_rejected"],
        shadow_patches_rejected=patch_stats["shadow_patches_rejected"],
        midtone_patch_count=patch_stats["midtone_patch_count"],
        patch_evidence=patch_stats["patch_evidence"] if not patch_fallback_used else [],
        warnings=warnings,
    )


def _region_consistency(regions: list) -> float:
    """Return a 0-1 consistency score based on pairwise Lab distance between
    the given regions' median colors. 1.0 = perfect agreement. Intended to
    be called with the regions actually used in the final combination, so
    it reflects the agreement of what was kept, not regions already
    excluded as contaminated."""
    labs = [np.array(r.median_lab) for r in regions if r.median_lab is not None]
    if len(labs) < 2:
        return 1.0 if labs else 0.0

    spread = float(np.mean(np.std(np.stack(labs), axis=0)))
    return float(np.clip(1.0 - spread / REGION_DISAGREEMENT_THRESHOLD, 0.0, 1.0))


def _lab_distance(a, b) -> float:
    _assert_skimage_lab_scale(np.array([a, b], dtype=np.float64))
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def _cheek_anchor_lab(reliable_by_name: dict):
    """Weighted-average Lab of whichever cheek region(s) are reliable.
    Cheeks are the least likely regions to be contaminated by hair or
    facial-hair shadow, so they serve as the trust anchor for judging the
    (optional) forehead and jawline regions. Returns None if no cheek is
    reliable."""
    cheek_regions = [reliable_by_name[n] for n in CHEEK_NAMES if n in reliable_by_name]
    if not cheek_regions:
        return None
    weights = np.array([r.valid_pixel_count for r in cheek_regions], dtype=np.float64)
    weights = weights / weights.sum()
    lab_stack = np.array([r.median_lab for r in cheek_regions], dtype=np.float64)
    return np.average(lab_stack, axis=0, weights=weights)


def _both_cheeks_agree(reliable_by_name: dict) -> bool:
    left = reliable_by_name.get("left_cheek")
    right = reliable_by_name.get("right_cheek")
    if left is None or right is None or left.median_lab is None or right.median_lab is None:
        return False
    return _lab_distance(left.median_lab, right.median_lab) <= CHEEK_AGREEMENT_LAB_DISTANCE


def _apply_forehead_and_jawline_rules(reliable_by_name: dict) -> list:
    """Mutate forehead/jawline RegionSkinResults in place against the cheek
    anchor color, and return any resulting warning strings.

    - Forehead: excluded outright if it disagrees strongly with the cheek
      tone (likely hair/fringe or shadow contamination).
    - Jawline: never excluded, but its combination weight is reduced when
      it is specifically darker than the cheeks (possible chin/neck shadow,
      contour, occlusion, or uneven lighting), since a jawline that is off-color in other ways
      may still carry useful skin-tone signal.

    No-ops (returns no warnings) if no cheek region is reliable, since
    there is then no trustworthy anchor to compare against.
    """
    warnings: list = []
    anchor_lab = _cheek_anchor_lab(reliable_by_name)
    if anchor_lab is None:
        return warnings

    for region in reliable_by_name.values():
        if region.makeup_influence_detected and not region.excluded:
            region.weight_multiplier *= MAKEUP_DOWNWEIGHT_FACTOR
            reason = "Possible makeup/highlight influence detected; region weight was reduced."
            region.downweight_reason = (
                f"{region.downweight_reason} {reason}" if region.downweight_reason else reason
            )
            warnings.append(f"{region.name.replace('_', ' ').title()}: {reason}")

        if region.reliability_score < REGION_RELIABILITY_THRESHOLD and not region.excluded:
            reason = (
                f"Region reliability is low ({region.reliability_score:.2f}) due to valid-pixel, "
                "stable-patch, variance, or shadow/highlight signals."
            )
            if region.name == FOREHEAD_NAME:
                region.excluded = True
                region.exclusion_reason = reason
            else:
                region.weight_multiplier *= LOW_RELIABILITY_DOWNWEIGHT_FACTOR
                region.downweight_reason = (
                    f"{region.downweight_reason} {reason}" if region.downweight_reason else reason
                )
            warnings.append(f"{region.name.replace('_', ' ').title()}: {reason}")

    forehead = reliable_by_name.get(FOREHEAD_NAME)
    if forehead is not None:
        if forehead.specular_highlight_detected and not forehead.excluded:
            forehead.weight_multiplier *= FOREHEAD_HIGHLIGHT_DOWNWEIGHT_FACTOR
            reason = (
                "Forehead showed possible specular highlight influence; its weight was strongly reduced "
                "so central-face shine does not make the foundation target too light."
            )
            forehead.downweight_reason = (
                f"{forehead.downweight_reason} {reason}" if forehead.downweight_reason else reason
            )
            warnings.append(forehead.downweight_reason)
        dist = _lab_distance(forehead.median_lab, anchor_lab)
        if dist > FOREHEAD_VS_CHEEK_OUTLIER_LAB_DISTANCE:
            forehead.excluded = True
            forehead.exclusion_reason = (
                f"Forehead color differs strongly from cheek tone (Lab distance {dist:.1f}); "
                "likely hair/fringe or shadow contamination, so it was excluded from the "
                "final skin tone estimate."
            )
            warnings.append(forehead.exclusion_reason)

    jawline = reliable_by_name.get(JAWLINE_NAME)
    if jawline is not None:
        delta_e = _lab_distance(jawline.median_lab, anchor_lab)
        darkness_gap = anchor_lab[0] - jawline.median_lab[0]
        contamination_concern = _jawline_has_contamination_concern(jawline)
        if delta_e > JAWLINE_QUALITY_DELTA_E_THRESHOLD and contamination_concern:
            jawline.weight_multiplier = JAWLINE_DOWNWEIGHT_FACTOR
            jawline.downweight_reason = (
                f"Jawline differs from cheek tone (Delta E {delta_e:.1f}, L difference {darkness_gap:.1f}); "
                "its weight in the final estimate was reduced because contamination signals were present "
                "(possible chin/neck shadow, "
                "contour, occlusion, or uneven lighting)."
            )
            warnings.append(jawline.downweight_reason)

    if _both_cheeks_agree(reliable_by_name):
        for name in (FOREHEAD_NAME, JAWLINE_NAME):
            region = reliable_by_name.get(name)
            if region is None or region.excluded:
                continue
            if region.weight_multiplier < 1.0:
                continue
            dist = _lab_distance(region.median_lab, anchor_lab)
            if dist > OPTIONAL_VS_CHEEK_DOWNWEIGHT_LAB_DISTANCE:
                if name == JAWLINE_NAME and not _jawline_has_contamination_concern(region):
                    continue
                region.weight_multiplier *= 0.5
                reason = (
                    f"{name.title()} differs from two agreeing cheeks (Lab distance {dist:.1f}); "
                    "it remains included as supporting signal with reduced weight."
                )
                region.downweight_reason = (
                    f"{region.downweight_reason} {reason}" if region.downweight_reason else reason
                )
                warnings.append(reason)

    return warnings


def _cheek_area_balance(region_results: dict) -> tuple[float, str | None]:
    """Return valid-area balance between cheeks and a gentle warning if poor."""
    left = region_results.get("left_cheek")
    right = region_results.get("right_cheek")
    if left is None or right is None:
        return 1.0, None
    if left.total_pixel_count < MIN_VALID_PIXELS_PER_REGION or right.total_pixel_count < MIN_VALID_PIXELS_PER_REGION:
        return 1.0, None

    smaller = min(left.valid_pixel_count, right.valid_pixel_count)
    larger = max(left.valid_pixel_count, right.valid_pixel_count)
    if larger == 0:
        return 1.0, None

    balance = smaller / larger
    if balance < CHEEK_AREA_IMBALANCE_WARNING_RATIO:
        smaller_name = "left cheek" if left.valid_pixel_count < right.valid_pixel_count else "right cheek"
        larger_name = "right cheek" if smaller_name == "left cheek" else "left cheek"
        return balance, (
            f"Cheek area imbalance detected: the {smaller_name} has much less valid skin area "
            f"than the {larger_name}. Confidence is reduced slightly, but both cheeks remain usable."
        )
    return balance, None


def _apply_cheek_visibility_weights(region_results: dict) -> list[str]:
    """Reduce a geometrically foreshortened cheek without excluding it.

    Total mask area is used rather than color, so valid darker skin is never
    treated as pose contamination merely because it has lower luminance.
    """
    left = region_results.get("left_cheek")
    right = region_results.get("right_cheek")
    if left is None or right is None:
        return []
    larger = max(left.total_pixel_count, right.total_pixel_count)
    if larger < MIN_VALID_PIXELS_PER_REGION:
        return []
    smaller = left if left.total_pixel_count < right.total_pixel_count else right
    area_ratio = smaller.total_pixel_count / max(larger, 1)
    if area_ratio >= CHEEK_FORESHORTENED_AREA_RATIO or not smaller.reliable:
        return []
    smaller.weight_multiplier *= CHEEK_FORESHORTENED_WEIGHT
    reason = (
        f"{smaller.name.replace('_', ' ').title()} mask area is only {area_ratio:.0%} "
        "of the opposite cheek, consistent with pose or partial visibility; "
        "it remains included with reduced influence."
    )
    smaller.downweight_reason = (
        f"{smaller.downweight_reason} {reason}" if smaller.downweight_reason else reason
    )
    return [reason]


def _jawline_has_contamination_concern(jawline: RegionSkinResult) -> bool:
    patch_lightness = [float(evidence.lab[0]) for evidence in jawline.patch_evidence]
    patch_lightness_span = (
        max(patch_lightness) - min(patch_lightness)
        if len(patch_lightness) >= 2
        else 0.0
    )
    return (
        jawline.lab_std > JAWLINE_HIGH_LAB_STD
        or jawline.valid_ratio < JAWLINE_LOW_VALID_RATIO
        or jawline.shadow_highlight_ratio > 0.25
        or patch_lightness_span > 8.0
        or jawline.reliability_score < REGION_RELIABILITY_THRESHOLD
        or jawline.specular_highlight_detected
    )


def _region_quality_label(score: float, excluded: bool = False) -> str:
    if excluded:
        return "excluded"
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "fair"
    return "poor"


def _region_role(region: RegionSkinResult, score: float) -> str:
    if region.excluded or not region.reliable or region.median_lab is None:
        return "excluded"
    if region.weight_multiplier < 1.0 or score < 55:
        return "reduced"
    if region.name in CHEEK_NAMES:
        return "trusted"
    return "supporting"


def _assign_region_quality(region_results: dict, reliable_by_name: dict) -> None:
    """Populate demo-facing per-region quality fields from existing extraction
    diagnostics. This is an interpretation layer only; it does not decide
    shade matching by itself.
    """
    anchor_lab = _cheek_anchor_lab(reliable_by_name)
    for region in region_results.values():
        reasons: list[str] = []
        quality_warnings: list[str] = []

        if not region.reliable or region.median_lab is None:
            score = min(region.valid_ratio * 40.0, 35.0)
            if region.total_pixel_count == 0:
                reasons.append("Region mask had no usable area.")
            else:
                reasons.append(
                    f"Only {region.valid_pixel_count}/{region.total_pixel_count} pixels survived skin filtering."
                )
            quality_warnings.extend(region.warnings)
        else:
            stable_patch_score = min(region.stable_patch_count / max(MIN_STABLE_PATCHES_PER_REGION, 1), 1.0)
            midtone_patch_score = min(region.midtone_patch_count / max(MIN_STABLE_PATCHES_PER_REGION, 1), 1.0)
            variance_score = float(np.clip(1.0 - region.lab_std / max(MAX_PATCH_LAB_STD, 1.0), 0.0, 1.0))
            usable_area_score = min(region.valid_pixel_count / max(MIN_VALID_PIXELS_PER_REGION * 2, 1), 1.0)
            shadow_highlight_score = float(np.clip(1.0 - region.shadow_highlight_ratio, 0.0, 1.0))
            patch_rejection_total = region.highlight_patches_rejected + region.shadow_patches_rejected
            patch_rejection_ratio = patch_rejection_total / max(
                patch_rejection_total + region.stable_patch_count + region.midtone_patch_count,
                1,
            )
            patch_rejection_score = float(np.clip(1.0 - patch_rejection_ratio, 0.0, 1.0))

            score = 100.0 * (
                0.28 * region.valid_ratio
                + 0.18 * stable_patch_score
                + 0.14 * midtone_patch_score
                + 0.16 * variance_score
                + 0.10 * usable_area_score
                + 0.08 * shadow_highlight_score
                + 0.06 * patch_rejection_score
            )

            if region.weight_multiplier < 1.0:
                score -= 14.0
                if region.downweight_reason:
                    quality_warnings.append(region.downweight_reason)
            if region.excluded:
                score = min(score, 30.0)
                if region.exclusion_reason:
                    quality_warnings.append(region.exclusion_reason)
            if region.specular_highlight_detected:
                score -= 8.0
                quality_warnings.append("Possible specular highlight influence detected.")
            if region.makeup_influence_detected:
                score -= 7.0
                quality_warnings.append("Possible makeup/highlight influence detected.")
            if region.shadow_highlight_ratio > 0.25:
                score -= 6.0
                quality_warnings.append("Elevated shadow/highlight rejection ratio.")
            if region.highlight_patches_rejected > 0:
                quality_warnings.append(f"{region.highlight_patches_rejected} highlight patch(es) rejected.")
            if region.shadow_patches_rejected > 0:
                quality_warnings.append(f"{region.shadow_patches_rejected} shadow patch(es) rejected.")

            reasons.append(f"Valid skin pixel ratio {region.valid_ratio:.0%}.")
            if region.stable_patch_count > 0:
                reasons.append(
                    f"{region.stable_patch_count} stable patch(es), {region.midtone_patch_count} mid-tone patch(es)."
                )
            elif region.patch_fallback_used:
                reasons.append("Used full-region median fallback because too few stable patches survived.")
            reasons.append(f"Local Lab variance {region.lab_std:.1f}.")
            if anchor_lab is not None and region.name not in CHEEK_NAMES:
                dist = _lab_distance(region.median_lab, anchor_lab)
                reasons.append(f"Distance from cheek anchor {dist:.1f} Delta E.")
            if region.name in CHEEK_NAMES:
                reasons.append("Primary cheek region for undertone and shade matching.")
            elif region.name == JAWLINE_NAME:
                if region.weight_multiplier >= 0.75 and not region.excluded:
                    reasons.append("Jawline/lower-cheek area supports shade depth when clean.")
                else:
                    reasons.append("Jawline is reduced only because contamination signals were present.")
            elif region.name == FOREHEAD_NAME:
                reasons.append("Forehead is treated as supporting evidence when reliable.")

        score = float(np.clip(score, 0.0, 100.0))
        role = _region_role(region, score)
        region.quality_score = score
        region.quality_label = _region_quality_label(score, role == "excluded")
        region.role = role
        region.quality_reasons = reasons
        region.quality_warnings = list(dict.fromkeys(quality_warnings))


def _region_base_weight(region_name: str) -> float:
    if region_name in CHEEK_NAMES:
        return 1.0
    if region_name == JAWLINE_NAME:
        return 0.75
    return OPTIONAL_REGION_BASE_WEIGHT


def _lab_to_rgb_tuple(lab: np.ndarray | tuple | list) -> tuple:
    lab_array = np.asarray(lab, dtype=np.float64).reshape(1, 1, 3)
    _assert_skimage_lab_scale(lab_array.reshape(1, 3))
    rgb = lab2rgb(lab_array).reshape(3)
    return tuple(np.round(np.clip(rgb * 255.0, 0, 255)).astype(int).tolist())


def _region_patch_role_weight(region: RegionSkinResult) -> float:
    if region.name in CHEEK_NAMES and region.role == "trusted":
        return 1.2
    if region.name == JAWLINE_NAME and region.role == "supporting":
        return 0.9
    if region.name == FOREHEAD_NAME:
        return 0.65
    if region.role == "reduced":
        return 0.55
    return 1.0


def _weighted_lab_mean(labs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(labs), dtype=np.float64)
    weights = weights / weights.sum()
    return np.average(labs, axis=0, weights=weights)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    if weights.sum() <= 0:
        weights = np.ones(len(values), dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()
    return float(sorted_values[np.searchsorted(cumulative, percentile / 100.0, side="left")])


def _ciede2000_matrix(labs: np.ndarray) -> np.ndarray:
    labs = np.asarray(labs, dtype=np.float64)
    first = labs[:, None, :]
    second = labs[None, :, :]
    return deltaE_ciede2000(
        np.broadcast_to(first, (len(labs), len(labs), 3)),
        np.broadcast_to(second, (len(labs), len(labs), 3)),
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return _weighted_percentile(values, weights, 50.0)


def _weighted_medoid_index(labs: np.ndarray, weights: np.ndarray) -> tuple[int, np.ndarray]:
    distances = _ciede2000_matrix(labs)
    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.sum() <= 0:
        normalized = np.ones(len(labs), dtype=np.float64)
    normalized = normalized / normalized.sum()
    costs = distances @ normalized
    return int(np.argmin(costs)), distances


def _normalize_patch_weights_by_region(
    candidates: list[dict],
    weights: np.ndarray,
) -> np.ndarray:
    """Normalize patch count within regions and enforce influence caps."""
    adjusted = np.asarray(weights, dtype=np.float64).copy()
    region_indices: dict[str, list[int]] = {}
    for idx, candidate in enumerate(candidates):
        region_indices.setdefault(candidate["region"], []).append(idx)

    for region, indices in region_indices.items():
        region = str(region)
        local = adjusted[indices]
        local_total = float(local.sum())
        if local_total <= 0:
            local = np.ones(len(indices), dtype=np.float64)
            local_total = float(local.sum())
        region_result = candidates[indices[0]]["region_result"]
        region_prior = (
            max(region_result.quality_score / 100.0, 0.05)
            * region_result.weight_multiplier
            * _region_base_weight(region)
            * _region_patch_role_weight(region_result)
        )
        adjusted[indices] = local / local_total * max(region_prior, 0.01)

    if adjusted.sum() <= 0:
        adjusted = np.ones(len(candidates), dtype=np.float64)
    adjusted /= adjusted.sum()

    # Iteratively cap an over-dominant region and redistribute its excess.
    for _ in range(4):
        changed = False
        for region, indices in region_indices.items():
            cap = REGION_CONTRIBUTION_CAPS.get(region, 0.45)
            total = float(adjusted[indices].sum())
            if total <= cap + 1e-9:
                continue
            changed = True
            excess = total - cap
            adjusted[indices] *= cap / total
            other = [i for i in range(len(adjusted)) if i not in indices]
            other_total = float(adjusted[other].sum())
            if other and other_total > 0:
                adjusted[other] += excess * adjusted[other] / other_total
        if not changed:
            break
    return adjusted / adjusted.sum()


def _foundation_target_lab_from_patches(candidates: list[dict], kept_indices: list[int], kept_weights: np.ndarray) -> np.ndarray:
    kept_labs = np.stack([candidates[idx]["lab"] for idx in kept_indices])
    base_lab = _weighted_lab_mean(kept_labs, kept_weights)

    cheek_indices = [
        idx
        for idx in kept_indices
        if candidates[idx]["region"] in CHEEK_NAMES and candidates[idx]["midtone"]
    ]
    if len(cheek_indices) >= 2:
        cheek_labs = np.stack([candidates[idx]["lab"] for idx in cheek_indices])
        cheek_weights = np.array([candidates[idx]["weight"] for idx in cheek_indices], dtype=np.float64)
        cheek_lab = _weighted_lab_mean(cheek_labs, cheek_weights)
        base_lab[1:] = cheek_lab[1:]

    depth_indices = [
        idx
        for idx in kept_indices
        if candidates[idx]["region"] in {*CHEEK_NAMES, JAWLINE_NAME} and candidates[idx]["midtone"]
    ]
    if len(depth_indices) >= 2:
        depth_labs = np.stack([candidates[idx]["lab"] for idx in depth_indices])
        depth_weights = np.array([candidates[idx]["weight"] for idx in depth_indices], dtype=np.float64)
        for pos, idx in enumerate(depth_indices):
            if candidates[idx]["region"] == JAWLINE_NAME:
                depth_weights[pos] *= 1.22
        lower_midtone_l = _weighted_percentile(depth_labs[:, 0], depth_weights, 42.0)
        weighted_l = float(_weighted_lab_mean(depth_labs, depth_weights)[0])
        # Use a lower-midtone L* target, but keep the shift conservative so
        # valid lighter diffuse cheek evidence is not ignored.
        base_lab[0] = min(base_lab[0], 0.65 * lower_midtone_l + 0.35 * weighted_l)

    return base_lab


def _aggregate_patch_candidates(combination_regions: list) -> tuple[tuple | None, tuple | None, dict]:
    candidates: list[dict] = []
    highlight_rejected = 0
    shadow_rejected = 0
    midtone_total = 0
    for region in combination_regions:
        highlight_rejected += region.highlight_patches_rejected
        shadow_rejected += region.shadow_patches_rejected
        midtone_total += region.midtone_patch_count
        evidence_items = region.patch_evidence
        if not evidence_items:
            evidence_items = [
                SkinPatchEvidence(
                    region=region.name,
                    rgb=rgb,
                    lab=lab,
                    quality=float(patch_quality),
                    is_midtone=bool(is_midtone),
                    lab_std=region.lab_std,
                    rgb_std=region.rgb_std,
                    luminance_median=float(lab[0]),
                    luminance_contrast=0.0,
                    highlight_ratio=0.0,
                    shadow_ratio=0.0,
                    top=0,
                    left=0,
                    size=PATCH_SIZE,
                )
                for rgb, lab, patch_quality, is_midtone in zip(
                    region.stable_patch_rgbs,
                    region.stable_patch_labs,
                    region.stable_patch_quality_scores,
                    region.stable_patch_midtone_flags,
                )
            ]
        for evidence in evidence_items:
            patch_quality = evidence.quality
            is_midtone = evidence.is_midtone
            weight = patch_quality * (1.12 if is_midtone else 0.82)
            candidates.append(
                {
                    "region": region.name,
                    "region_result": region,
                    "evidence": evidence,
                    "rgb": np.array(evidence.rgb, dtype=np.float64),
                    "lab": np.array(evidence.lab, dtype=np.float64),
                    "patch_quality": float(patch_quality),
                    "region_quality": float(max(region.quality_score / 100.0, 0.05)),
                    "weight": float(weight),
                    "midtone": bool(is_midtone),
                }
            )

    diagnostics = {
        "used": False,
        "stable_patches_available": len(candidates),
        "stable_patches_used": 0,
        "outlier_patches_rejected": 0,
        "highlight_patches_rejected": highlight_rejected,
        "shadow_patches_rejected": shadow_rejected,
        "midtone_patches_used": 0,
        "dominant_region_contribution": "none",
        "foundation_depth_strategy": "region fallback",
        "consensus_method": "region fallback",
        "consensus_medoid_lab": None,
        "outlier_threshold_delta_e": 0.0,
        "region_contributions": {},
        "adaptive_patch_sizes": sorted(
            {evidence.size for region in combination_regions for evidence in region.patch_evidence}
        ),
        "fallback_reason": "",
    }
    represented_regions = {c["region"] for c in candidates}
    if len(candidates) < 3 or len(represented_regions) < 2:
        diagnostics["fallback_reason"] = (
            "Patch voting fallback used because fewer than three stable patches across two regions were available."
        )
        return None, None, diagnostics

    labs = np.stack([c["lab"] for c in candidates])
    weights = _normalize_patch_weights_by_region(
        candidates,
        np.array([c["weight"] for c in candidates], dtype=np.float64),
    )
    medoid_idx, distance_matrix = _weighted_medoid_index(labs, weights)
    central_lab = labs[medoid_idx]
    distances = distance_matrix[medoid_idx]
    distance_median = _weighted_median(distances, weights)
    distance_mad = _weighted_median(np.abs(distances - distance_median), weights)
    robust_scale = max(1.4826 * distance_mad, 1.0)
    outlier_threshold = float(np.clip(distance_median + 3.0 * robust_scale, 6.0, 14.0))
    soft_threshold = float(np.clip(distance_median + 1.5 * robust_scale, 4.0, outlier_threshold))

    kept_indices: list[int] = []
    adjusted_weights = weights.copy()
    for idx, candidate in enumerate(candidates):
        distance = float(distances[idx])
        candidate_threshold = outlier_threshold
        if candidate["region"] == JAWLINE_NAME and candidate["midtone"]:
            candidate_threshold = min(outlier_threshold + 8.0, 22.0)
        if distance > candidate_threshold:
            diagnostics["outlier_patches_rejected"] += 1
            continue
        if distance > soft_threshold:
            adjusted_weights[idx] *= max(
                0.2,
                1.0 - (distance - soft_threshold) / max(candidate_threshold - soft_threshold, 1.0),
            )
        kept_indices.append(idx)

    if len(kept_indices) < 3 or len({candidates[i]["region"] for i in kept_indices}) < 2:
        diagnostics["fallback_reason"] = (
            "Patch voting fallback used because robust outlier filtering left too few stable patches."
        )
        return None, None, diagnostics

    kept_labs = labs[kept_indices]
    kept_weights = adjusted_weights[kept_indices]
    kept_distances = distances[kept_indices]
    trim_cutoff = _weighted_percentile(kept_distances, kept_weights, 90.0)
    trim_mask = kept_distances <= max(trim_cutoff, 6.0)
    if trim_mask.sum() >= 3 and len(set(candidates[i]["region"] for i, keep in zip(kept_indices, trim_mask) if keep)) >= 2:
        kept_indices = [idx for idx, keep in zip(kept_indices, trim_mask) if keep]
        kept_labs = labs[kept_indices]
        kept_weights = adjusted_weights[kept_indices]

    measured_lab_array = _weighted_lab_mean(kept_labs, kept_weights)
    target_lab_array = _foundation_target_lab_from_patches(candidates, kept_indices, kept_weights)
    final_lab_array = measured_lab_array
    final_rgb = _lab_to_rgb_tuple(final_lab_array)
    region_weight_totals: dict[str, float] = {}
    for idx, weight in zip(kept_indices, kept_weights):
        region_weight_totals[candidates[idx]["region"]] = region_weight_totals.get(candidates[idx]["region"], 0.0) + float(weight)
    dominant_region = max(region_weight_totals, key=region_weight_totals.get)
    total_weight = sum(region_weight_totals.values()) or 1.0

    diagnostics.update(
        {
            "used": True,
            "stable_patches_used": len(kept_indices),
            "midtone_patches_used": sum(1 for idx in kept_indices if candidates[idx]["midtone"]),
            "dominant_region_contribution": (
                f"{dominant_region.replace('_', ' ').title()} ({region_weight_totals[dominant_region] / total_weight:.0%})"
            ),
            "foundation_depth_strategy": (
                "undertone from diffuse cheek patches; depth L* from reliable lower-midtone cheek/jawline patches"
            ),
            "consensus_method": "weighted CIEDE2000 medoid with MAD outlier rejection",
            "consensus_medoid_lab": tuple(central_lab.tolist()),
            "outlier_threshold_delta_e": outlier_threshold,
            "patch_foundation_target_lab": tuple(target_lab_array.tolist()),
            "patch_foundation_target_rgb": _lab_to_rgb_tuple(target_lab_array),
            "retained_patch_labs": [
                tuple(candidates[idx]["lab"].tolist()) for idx in kept_indices
            ],
            "retained_patch_weights": [float(weight) for weight in kept_weights],
            "retained_patch_regions": [
                str(candidates[idx]["region"]) for idx in kept_indices
            ],
        }
    )
    diagnostics["region_contributions"] = {
        region: float(weight / total_weight)
        for region, weight in region_weight_totals.items()
    }
    return final_rgb, tuple(final_lab_array.tolist()), diagnostics


def _bootstrap_patch_uncertainty(
    patch_diagnostics: dict,
    reference_lab: tuple,
) -> tuple[list[tuple], dict]:
    labs_raw = patch_diagnostics.get("retained_patch_labs", [])
    weights_raw = patch_diagnostics.get("retained_patch_weights", [])
    regions = patch_diagnostics.get("retained_patch_regions", [])
    fallback = {
        "bootstrap_iterations": 0,
        "lab_std": (0.0, 0.0, 0.0),
        "lab_interval_90": {},
        "l_interval_90": None,
        "delta_e_radius_p90": 12.0,
        "patch_agreement": 0.0,
        "stability_score": 45.0,
        "method": "insufficient stable patches",
    }
    if len(labs_raw) < 3 or len(set(regions)) < 2:
        return [], fallback

    labs = np.asarray(labs_raw, dtype=np.float64)
    weights = np.asarray(weights_raw, dtype=np.float64)
    if len(weights) != len(labs) or weights.sum() <= 0:
        weights = np.ones(len(labs), dtype=np.float64)
    region_indices = {
        region: np.array([idx for idx, value in enumerate(regions) if value == region], dtype=int)
        for region in dict.fromkeys(regions)
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample_indices = []
        for indices in region_indices.values():
            sample_indices.extend(
                rng.choice(indices, size=len(indices), replace=True).tolist()
            )
        sample_indices_array = np.asarray(sample_indices, dtype=int)
        samples.append(
            _weighted_lab_mean(
                labs[sample_indices_array],
                weights[sample_indices_array],
            )
        )

    sample_array = np.stack(samples)
    reference = np.asarray(reference_lab, dtype=np.float64)
    reference_tiled = np.tile(reference, (len(sample_array), 1))
    radii = deltaE_ciede2000(reference_tiled, sample_array)
    lower = np.percentile(sample_array, 5, axis=0)
    upper = np.percentile(sample_array, 95, axis=0)
    radius_p90 = float(np.percentile(radii, 90))
    patch_agreement = float(np.clip(1.0 - radius_p90 / 12.0, 0.0, 1.0))
    diagnostics = {
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "lab_std": tuple(np.std(sample_array, axis=0).tolist()),
        "lab_interval_90": {
            channel: (float(lower[idx]), float(upper[idx]))
            for idx, channel in enumerate(("l", "a", "b"))
        },
        "l_interval_90": (float(lower[0]), float(upper[0])),
        "delta_e_radius_p90": radius_p90,
        "patch_agreement": patch_agreement,
        "stability_score": float(np.clip(45.0 + 55.0 * patch_agreement, 0.0, 100.0)),
        "method": "96 deterministic stratified patch bootstrap samples",
    }
    return [tuple(sample.tolist()) for sample in sample_array], diagnostics


def _shift_bootstrap_to_target(
    bootstrap_labs: list[tuple],
    measured_lab: tuple,
    target_lab: tuple,
) -> list[tuple]:
    if not bootstrap_labs:
        return []
    samples = np.asarray(bootstrap_labs, dtype=np.float64)
    offset = np.asarray(target_lab, dtype=np.float64) - np.asarray(measured_lab, dtype=np.float64)
    shifted = samples + offset
    shifted[:, 0] = np.clip(shifted[:, 0], 0.0, 100.0)
    return [tuple(sample.tolist()) for sample in shifted]


def _weighted_region_blend(combination_regions: list) -> tuple[tuple, tuple]:
    weights = np.array(
        [
            r.valid_pixel_count * r.weight_multiplier * _region_base_weight(r.name)
            for r in combination_regions
        ],
        dtype=np.float64,
    )
    if weights.sum() <= 0:
        weights = np.ones(len(combination_regions), dtype=np.float64)
    weights = weights / weights.sum()

    rgb_stack = np.array([r.median_rgb for r in combination_regions], dtype=np.float64)
    lab_stack = np.array([r.median_lab for r in combination_regions], dtype=np.float64)
    blended_rgb = tuple(np.round(np.average(rgb_stack, axis=0, weights=weights)).astype(int).tolist())
    blended_lab = tuple(np.average(lab_stack, axis=0, weights=weights).tolist())
    return blended_rgb, blended_lab


def _final_color_from_regions(combination_regions: list) -> tuple[tuple, tuple, dict]:
    region_blend_rgb, region_blend_lab = _weighted_region_blend(combination_regions)
    patch_rgb, patch_lab, patch_voting_diagnostics = _aggregate_patch_candidates(combination_regions)
    if patch_rgb is not None and patch_lab is not None:
        return patch_rgb, patch_lab, patch_voting_diagnostics
    return region_blend_rgb, region_blend_lab, patch_voting_diagnostics


def _stability_label(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "fair"
    return "poor"


def _region_stability_summary(label: str, most_influential_region: str | None) -> str:
    if label in {"excellent", "good"}:
        return (
            f"Region stability was {label}; removing any one trusted region did not "
            "significantly change the final tone."
        )
    region_text = (
        most_influential_region.replace("_", " ").title()
        if most_influential_region
        else "one region"
    )
    return (
        f"Region stability was {label}; {region_text} had stronger influence, "
        "so confidence was reduced."
    )


def _analyze_region_stability(combination_regions: list, final_lab: tuple) -> dict:
    diagnostics = {
        "stability_score": 100.0,
        "stability_label": "excellent",
        "most_influential_region": "none",
        "unstable_regions": [],
        "leave_one_out_delta_e": {},
        "warnings": [],
        "reasons": [],
        "summary": "Region stability was excellent; removing any one trusted region did not significantly change the final tone.",
    }
    if len(combination_regions) < 3:
        diagnostics.update(
            {
                "stability_score": 70.0 if len(combination_regions) == 2 else 55.0,
                "stability_label": "good" if len(combination_regions) == 2 else "fair",
                "summary": "Region stability was limited because fewer than three usable regions were available.",
                "reasons": ["Fewer than three usable regions limited leave-one-region-out analysis."],
            }
        )
        return diagnostics

    full_lab = np.array(final_lab, dtype=np.float64)
    deltas: dict[str, float] = {}
    for region in combination_regions:
        remaining = [r for r in combination_regions if r.name != region.name]
        if not remaining:
            continue
        _, subset_lab, _ = _final_color_from_regions(remaining)
        deltas[region.name] = _lab_distance(full_lab, subset_lab)

    if not deltas:
        return diagnostics

    max_delta = max(deltas.values())
    avg_delta = float(np.mean(list(deltas.values())))
    most_influential_region = max(deltas, key=deltas.get)
    unstable_regions = [name for name, delta in deltas.items() if delta >= 8.0]
    score = float(np.clip(100.0 - max_delta * 5.5 - avg_delta * 1.5, 0.0, 100.0))
    label = _stability_label(score)
    summary = _region_stability_summary(label, most_influential_region)
    warnings = []
    if unstable_regions or label in {"fair", "poor"}:
        warnings.append(summary)

    diagnostics.update(
        {
            "stability_score": score,
            "stability_label": label,
            "most_influential_region": most_influential_region,
            "unstable_regions": unstable_regions,
            "leave_one_out_delta_e": {name: round(delta, 2) for name, delta in deltas.items()},
            "warnings": warnings,
            "reasons": [
                f"Maximum leave-one-region-out shift was {max_delta:.1f} Delta E.",
                f"Average leave-one-region-out shift was {avg_delta:.1f} Delta E.",
            ],
            "summary": summary,
        }
    )
    return diagnostics


def _region_has_highlight_bias(region: RegionSkinResult) -> bool:
    return (
        region.specular_highlight_detected
        or region.highlight_patches_rejected > 0
        or region.makeup_influence_detected
    )


def _lower_face_depth_evidence(region_results: dict) -> tuple[float | None, dict]:
    candidates = []
    region_lightness: dict[str, float] = {}
    for name in (*CHEEK_NAMES, JAWLINE_NAME):
        region = region_results.get(name)
        if region is None or not region.reliable or region.excluded or region.median_lab is None:
            continue
        if name == JAWLINE_NAME and _jawline_has_contamination_concern(region):
            continue
        labs = [
            np.array(lab, dtype=np.float64)
            for lab, is_midtone in zip(region.stable_patch_labs, region.stable_patch_midtone_flags)
            if is_midtone
        ]
        weights = [
            float(score)
            for score, is_midtone in zip(region.stable_patch_quality_scores, region.stable_patch_midtone_flags)
            if is_midtone
        ]
        if not labs:
            labs = [np.array(region.median_lab, dtype=np.float64)]
            weights = [max(region.quality_score / 100.0, 0.2)]
        for lab, weight in zip(labs, weights):
            if name == JAWLINE_NAME:
                weight *= 1.25
            candidates.append((name, lab, weight))
        region_lightness[name] = float(
            _weighted_percentile(
                np.array([lab[0] for lab in labs], dtype=np.float64),
                np.array(weights, dtype=np.float64),
                40.0,
            )
        )

    represented_regions = list(dict.fromkeys([name for name, _, _ in candidates]))
    region_l_span = (
        max(region_lightness.values()) - min(region_lightness.values())
        if len(region_lightness) >= 2
        else 0.0
    )
    region_agreement = len(region_lightness) >= 2 and region_l_span <= 10.0
    diagnostics = {
        "lower_face_patch_count": len(candidates),
        "lower_face_regions": represented_regions,
        "lower_face_region_lightness": region_lightness,
        "lower_face_region_l_span": region_l_span,
        "lower_face_region_agreement": region_agreement,
    }
    if len(candidates) < 2 or not region_agreement:
        return None, diagnostics
    labs = np.stack([lab for _, lab, _ in candidates])
    weights = np.array([weight for _, _, weight in candidates], dtype=np.float64)
    return _weighted_percentile(labs[:, 0], weights, 40.0), diagnostics


def _build_foundation_target(
    measured_rgb: tuple,
    measured_lab: tuple,
    region_results: dict,
    patch_voting_diagnostics: dict,
    depth_estimate: str | None,
) -> tuple[tuple, tuple, bool, str, dict]:
    measured_lab_array = np.array(measured_lab, dtype=np.float64)
    measured_rgb_tuple = tuple(int(v) for v in measured_rgb)
    diagnostics = {
        "criteria": {
            "highlight_influence": False,
            "measured_brighter_than_lower": False,
            "lower_face_reliable": False,
        },
        "measured_lab": tuple(measured_lab_array.tolist()),
    }

    highlight_regions = [
        name for name, region in region_results.items() if _region_has_highlight_bias(region)
    ]
    diagnostics["highlight_regions"] = highlight_regions
    diagnostics["criteria"]["highlight_influence"] = bool(highlight_regions)

    lower_l, lower_diag = _lower_face_depth_evidence(region_results)
    diagnostics.update(lower_diag)
    diagnostics["lower_face_depth_l"] = lower_l
    diagnostics["criteria"]["lower_face_reliable"] = False

    central_regions = [
        region
        for name, region in region_results.items()
        if name in (FOREHEAD_NAME, *CHEEK_NAMES) and region.median_lab is not None and region.reliable
    ]
    central_l = float(np.mean([region.median_lab[0] for region in central_regions])) if central_regions else measured_lab_array[0]
    diagnostics["central_face_l"] = central_l
    if lower_l is not None:
        diagnostics["central_minus_lower_l"] = central_l - lower_l
        diagnostics["measured_minus_lower_l"] = float(measured_lab_array[0] - lower_l)
        diagnostics["criteria"]["measured_brighter_than_lower"] = bool(
            (measured_lab_array[0] - lower_l) >= 3.0
        )
        diagnostics["criteria"]["lower_face_reliable"] = bool(
            diagnostics["criteria"]["measured_brighter_than_lower"]
        )
    else:
        diagnostics["central_minus_lower_l"] = 0.0
        diagnostics["measured_minus_lower_l"] = 0.0

    target_lab = measured_lab_array.copy()
    patch_target = patch_voting_diagnostics.get("patch_foundation_target_lab")
    if patch_target is not None:
        patch_target_lab = np.array(patch_target, dtype=np.float64)
        target_lab[1:] = 0.25 * measured_lab_array[1:] + 0.75 * patch_target_lab[1:]

    all_criteria = all(diagnostics["criteria"].values())
    if not all_criteria:
        reason = "Foundation target matches measured visible tone; depth-safe adjustment criteria were not all met."
        diagnostics["active"] = False
        diagnostics["reason"] = reason
        return measured_rgb_tuple, tuple(measured_lab_array.tolist()), False, reason, diagnostics

    supported_gap = float(measured_lab_array[0] - lower_l)
    max_shift = (
        5.0
        if depth_estimate == "rich-deep"
        else 4.0
        if depth_estimate in {"tan", "deep"}
        else 3.0
    )
    supported_shift = min(max_shift, 0.60 * supported_gap)
    target_l = float(measured_lab_array[0]) - supported_shift
    if float(measured_lab_array[0]) - target_l < 1.0:
        reason = "Foundation target matches measured visible tone; lower-face evidence did not support a meaningful deeper target."
        diagnostics["active"] = False
        diagnostics["reason"] = reason
        return measured_rgb_tuple, tuple(measured_lab_array.tolist()), False, reason, diagnostics

    target_lab[0] = target_l
    target_lab_tuple = tuple(target_lab.tolist())
    target_rgb = _lab_to_rgb_tuple(target_lab)
    reason = (
        f"Foundation target L* was adjusted slightly deeper by {supported_shift:.1f} because highlight "
        "influence was detected and at least two agreeing lower-face regions supported "
        "a deeper base tone; cheek-derived undertone was preserved."
    )
    diagnostics.update(
        {
            "active": True,
            "reason": reason,
            "target_lab": target_lab_tuple,
            "target_rgb": target_rgb,
            "l_adjustment": float(measured_lab_array[0] - target_l),
            "maximum_l_adjustment": max_shift,
            "supported_l_gap": supported_gap,
        }
    )
    return target_rgb, target_lab_tuple, True, reason, diagnostics


def _adjusted_region_consistency(regions: list, reliable_by_name: dict) -> float:
    consistency = _region_consistency(regions)
    if not _both_cheeks_agree(reliable_by_name):
        return consistency
    cheek_regions = [reliable_by_name[n] for n in CHEEK_NAMES if n in reliable_by_name]
    cheek_consistency = _region_consistency(cheek_regions)
    return float(max(consistency, cheek_consistency * 0.9))


def _build_extraction_quality_reasons(region_results: dict, skin_result_fields: dict) -> list:
    reasons = []
    included = skin_result_fields.get("included_region_names", [])
    excluded = skin_result_fields.get("excluded_region_names", [])
    reduced = [
        name
        for name, region in region_results.items()
        if not region.excluded and region.weight_multiplier < 1.0
    ]

    reasons.append(
        "Included regions: "
        + (", ".join(n.replace("_", " ").title() for n in included) if included else "None")
        + "."
    )
    not_used = [
        name
        for name, region in region_results.items()
        if name not in included and (region.excluded or not region.reliable)
    ]
    if not_used:
        reasons.append(
            "Not-used regions: "
            + ", ".join(n.replace("_", " ").title() for n in not_used)
            + "."
        )
    if reduced:
        reasons.append(
            "Reduced-weight regions: "
            + ", ".join(n.replace("_", " ").title() for n in reduced)
            + "."
        )
    for name, region in region_results.items():
        if region.stable_patch_count > 0:
            reasons.append(
                f"{name.replace('_', ' ').title()}: used {region.stable_patch_count} stable patch(es)."
            )
        if region.midtone_patch_count > 0:
            reasons.append(
                f"{name.replace('_', ' ').title()}: used {region.midtone_patch_count} diffuse mid-tone patch(es)."
            )
        if region.highlight_patches_rejected > 0:
            reasons.append(
                "Bright highlight patches were excluded so the recommendation is based on diffuse skin tone rather than shine."
            )
        if region.specular_highlight_detected:
            reasons.append(f"{name.replace('_', ' ').title()}: possible specular highlight influence detected.")
        if region.patch_fallback_used and region.median_rgb is not None:
            reasons.append(
                f"{name.replace('_', ' ').title()}: used full-region median fallback."
            )
        if region.status_reason:
            reasons.append(f"{name.replace('_', ' ').title()}: {region.status_reason}")
        reasons.append(
            f"{name.replace('_', ' ').title()}: reliability score {region.reliability_score:.0%}."
        )

    if skin_result_fields.get("usable_region_count", 0) < 3:
        reasons.append("Fewer than 3 regions were usable, so extraction confidence is reduced slightly.")
    if skin_result_fields.get("cheek_area_balance", 1.0) < CHEEK_AREA_IMBALANCE_WARNING_RATIO:
        reasons.append("Cheek valid-area imbalance reduced confidence slightly.")
    jawline = region_results.get(JAWLINE_NAME)
    if jawline is not None and jawline.reliable and not jawline.excluded and jawline.weight_multiplier >= 0.75:
        reasons.append("Jawline/lower-cheek patches supported the shade depth estimate.")
    return reasons


def extract_skin_tone(image_rgb: np.ndarray, masks: dict) -> SkinToneResult:
    """Extract a representative skin color from masked face regions.

    For each of forehead/left_cheek/right_cheek/jawline: filters out
    shadow/highlight luminance extremes and extreme-saturation pixels, then
    takes the median RGB/Lab. Cheeks anchor the trust check: forehead is
    excluded outright if it disagrees strongly with the cheeks (likely
    hair/shadow contamination); jawline is down-weighted, not excluded,
    when it is specifically darker than the cheeks (possible chin/neck shadow,
    contour, occlusion, or uneven lighting). The remaining (non-excluded) reliable regions are
    combined, weighted by valid pixel count and any down-weighting, into
    one final skin color.
    """
    warnings: list = []
    region_results = {}
    for name in REGION_NAMES:
        mask = masks.get(name)
        if mask is None:
            continue
        region_results[name] = _extract_region(image_rgb, mask, name)
        warnings.extend(region_results[name].warnings)

    reliable_by_name = {name: r for name, r in region_results.items() if r.reliable}
    warnings.extend(_apply_forehead_and_jawline_rules(reliable_by_name))
    warnings.extend(_apply_cheek_visibility_weights(region_results))
    cheek_area_balance, cheek_area_warning = _cheek_area_balance(region_results)
    if cheek_area_warning:
        warnings.append(cheek_area_warning)
    _assign_region_quality(region_results, reliable_by_name)

    combination_regions = [r for r in reliable_by_name.values() if not r.excluded]

    if not combination_regions:
        # Fall back to any region with at least some valid pixels (and not
        # explicitly excluded) so the app can still show a (low-confidence)
        # result instead of failing outright.
        fallback_regions = [
            r for r in region_results.values() if r.median_rgb is not None and not r.excluded
        ]
        if not fallback_regions:
            warnings.append(
                "Could not extract a reliable skin color from any region. "
                "Try a clearer, well-lit, front-facing photo."
            )
            return SkinToneResult(
                rgb=(0, 0, 0),
                lab=(0.0, 0.0, 0.0),
                region_results=region_results,
                quality_score=0.0,
                region_consistency=0.0,
                avg_valid_pixel_ratio=0.0,
                cheek_area_balance=cheek_area_balance,
                usable_region_count=0,
                included_region_names=[],
                excluded_region_names=[n for n, r in region_results.items() if r.excluded],
                extraction_quality_reasons=[],
                depth_estimate=None,
                ita_degrees=None,
                ita_category=None,
                warnings=warnings,
                success=False,
            )
        warnings.append(
            "No region met the minimum valid-pixel threshold; using best-available regions "
            "with reduced confidence."
        )
        combination_regions = fallback_regions

    final_rgb, final_lab, patch_voting_diagnostics = _final_color_from_regions(combination_regions)
    if not patch_voting_diagnostics.get("used") and patch_voting_diagnostics.get("fallback_reason"):
        warnings.append(patch_voting_diagnostics["fallback_reason"])
    measured_bootstrap_labs, uncertainty_diagnostics = _bootstrap_patch_uncertainty(
        patch_voting_diagnostics,
        final_lab,
    )
    depth_diagnostic = build_skin_depth_diagnostic(final_lab)
    (
        foundation_target_rgb,
        foundation_target_lab,
        foundation_target_active,
        foundation_target_reason,
        foundation_target_diagnostics,
    ) = _build_foundation_target(
        final_rgb,
        final_lab,
        region_results,
        patch_voting_diagnostics,
        depth_diagnostic.depth_category,
    )
    bootstrap_labs = _shift_bootstrap_to_target(
        measured_bootstrap_labs,
        final_lab,
        foundation_target_lab,
    )
    if foundation_target_active:
        warnings.append(foundation_target_reason)
    stability_diagnostics = _analyze_region_stability(combination_regions, final_lab)
    warnings.extend(stability_diagnostics.get("warnings", []))

    consistency = _adjusted_region_consistency(combination_regions, reliable_by_name)
    avg_valid_ratio = float(np.mean([r.valid_ratio for r in region_results.values()])) if region_results else 0.0

    if consistency < 0.5:
        warnings.append(
            "Skin regions disagree noticeably in color (possible uneven lighting or shadows)."
        )

    included_region_names = [r.name for r in combination_regions]
    excluded_region_names = [name for name, r in region_results.items() if r.excluded]
    usable_region_count = len(combination_regions)
    if usable_region_count < 3:
        warnings.append("Fewer than 3 skin regions were usable; extraction confidence is reduced slightly.")

    region_count_score = min(usable_region_count / 3.0, 1.0)
    quality_score = float(
        np.clip(0.45 * consistency + 0.4 * avg_valid_ratio + 0.15 * region_count_score, 0.0, 1.0)
    )
    stability_score = float(stability_diagnostics.get("stability_score", 100.0)) / 100.0
    if stability_score < 0.85:
        quality_score = float(np.clip(quality_score * (0.92 + 0.08 * stability_score), 0.0, 1.0))
    quality_fields = {
        "included_region_names": included_region_names,
        "excluded_region_names": excluded_region_names,
        "usable_region_count": usable_region_count,
        "cheek_area_balance": cheek_area_balance,
    }
    extraction_quality_reasons = _build_extraction_quality_reasons(region_results, quality_fields)
    if patch_voting_diagnostics.get("used"):
        extraction_quality_reasons.append(
            "Final skin tone was aggregated from stable diffuse patches across trusted regions."
        )
    elif patch_voting_diagnostics.get("fallback_reason"):
        extraction_quality_reasons.append(patch_voting_diagnostics["fallback_reason"])
    extraction_quality_reasons.append(stability_diagnostics["summary"])
    extraction_quality_reasons.append(foundation_target_reason)

    return SkinToneResult(
        rgb=final_rgb,
        lab=final_lab,
        region_results=region_results,
        quality_score=quality_score,
        region_consistency=consistency,
        avg_valid_pixel_ratio=avg_valid_ratio,
        cheek_area_balance=cheek_area_balance,
        usable_region_count=usable_region_count,
        included_region_names=included_region_names,
        excluded_region_names=excluded_region_names,
        extraction_quality_reasons=extraction_quality_reasons,
        patch_voting_diagnostics=patch_voting_diagnostics,
        stability_diagnostics=stability_diagnostics,
        foundation_target_rgb=foundation_target_rgb,
        foundation_target_lab=foundation_target_lab,
        foundation_target_active=foundation_target_active,
        foundation_target_reason=foundation_target_reason,
        foundation_target_diagnostics=foundation_target_diagnostics,
        bootstrap_labs=bootstrap_labs,
        uncertainty_diagnostics=uncertainty_diagnostics,
        depth_estimate=depth_diagnostic.depth_category,
        ita_degrees=depth_diagnostic.ita_degrees,
        ita_category=depth_diagnostic.ita_category,
        warnings=warnings,
        success=True,
    )
