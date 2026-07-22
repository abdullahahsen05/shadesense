"""Representative skin color extraction and filtering from masked face regions."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.color import rgb2lab

REGION_NAMES = ["forehead", "left_cheek", "right_cheek", "jawline"]
CHEEK_NAMES = ("left_cheek", "right_cheek")
FOREHEAD_NAME = "forehead"
JAWLINE_NAME = "jawline"

LUMINANCE_LOWER_PERCENTILE = 10
LUMINANCE_UPPER_PERCENTILE = 90
MIN_ABSOLUTE_LUMINANCE = 12.0
MAX_ABSOLUTE_LUMINANCE = 96.0
SATURATION_UPPER_PERCENTILE = 92
MAX_ABSOLUTE_SATURATION = 170.0
MIN_VALID_PIXELS_PER_REGION = 100
REGION_DISAGREEMENT_THRESHOLD = 12.0  # Lab distance considered "high disagreement"
CHEEK_AGREEMENT_LAB_DISTANCE = 8.0
OPTIONAL_VS_CHEEK_DOWNWEIGHT_LAB_DISTANCE = 14.0
OPTIONAL_REGION_BASE_WEIGHT = 0.55
PATCH_SIZE = 18
PATCH_STRIDE = 12
MIN_VALID_PIXELS_PER_PATCH = 45
MIN_STABLE_PATCHES_PER_REGION = 2
MAX_STABLE_PATCHES_PER_REGION = 8
MAX_PATCH_LAB_STD = 8.0
MAX_PATCH_RGB_STD = 34.0

# Forehead is useful but optional: if it disagrees strongly with the cheek
# tone (full Lab distance), it is excluded outright — likely hair/fringe or
# shadow contamination rather than skin.
FOREHEAD_VS_CHEEK_OUTLIER_LAB_DISTANCE = 20.0

# Jawline is never excluded outright, but facial hair or chin/neck shadow
# tends to make it specifically darker (not just differently colored) than
# the cheeks, so a lightness (L*) gap beyond this threshold reduces —
# rather than removes — its weight in the final combination.
JAWLINE_DARKNESS_L_THRESHOLD = 10.0
JAWLINE_DOWNWEIGHT_FACTOR = 0.35
CHEEK_AREA_IMBALANCE_WARNING_RATIO = 0.45


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
    patch_fallback_used: bool = False
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
    warnings: list = field(default_factory=list)
    success: bool = True


def _to_lab(pixels_rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) uint8 RGB pixel array to (N, 3) Lab floats."""
    return rgb2lab(pixels_rgb.reshape(-1, 1, 3).astype(np.float64) / 255.0).reshape(-1, 3)


def _to_saturation(pixels_rgb: np.ndarray) -> np.ndarray:
    """Return the HSV saturation channel (0-255) for an (N, 3) uint8 RGB pixel array."""
    hsv = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    return hsv.reshape(-1, 3)[:, 1].astype(np.float64)


def _filter_skin_pixels(pixels_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively filter masked pixels down to plausible skin pixels."""
    if len(pixels_rgb) == 0:
        return pixels_rgb, np.empty((0, 3), dtype=np.float64)

    lab = _to_lab(pixels_rgb)
    saturation = _to_saturation(pixels_rgb)
    luminance = lab[:, 0]

    lum_low = max(
        float(np.percentile(luminance, LUMINANCE_LOWER_PERCENTILE)),
        MIN_ABSOLUTE_LUMINANCE,
    )
    lum_high = min(
        float(np.percentile(luminance, LUMINANCE_UPPER_PERCENTILE)),
        MAX_ABSOLUTE_LUMINANCE,
    )
    if lum_low >= lum_high:
        lum_low, lum_high = float(np.min(luminance)), float(np.max(luminance))

    sat_high = min(
        float(np.percentile(saturation, SATURATION_UPPER_PERCENTILE)),
        MAX_ABSOLUTE_SATURATION,
    )

    red_dominance = (
        (pixels_rgb[:, 0].astype(np.int16) - pixels_rgb[:, 1].astype(np.int16) > 48)
        & (pixels_rgb[:, 0].astype(np.int16) - pixels_rgb[:, 2].astype(np.int16) > 48)
        & (saturation > 115)
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
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return RGB/Lab medians from the most stable patches in a mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), 0

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    stable = []

    for top in range(y0, max(y0 + 1, y1 - PATCH_SIZE + 1), PATCH_STRIDE):
        for left in range(x0, max(x0 + 1, x1 - PATCH_SIZE + 1), PATCH_STRIDE):
            patch_mask = mask[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
            patch_pixels = image_rgb[top : top + PATCH_SIZE, left : left + PATCH_SIZE][patch_mask > 0]
            if len(patch_pixels) < MIN_VALID_PIXELS_PER_PATCH:
                continue

            valid_rgb, valid_lab = _filter_skin_pixels(patch_pixels)
            if len(valid_rgb) < MIN_VALID_PIXELS_PER_PATCH:
                continue

            lab_std = float(np.mean(np.std(valid_lab, axis=0)))
            rgb_std = float(np.mean(np.std(valid_rgb.astype(np.float64), axis=0)))
            if lab_std > MAX_PATCH_LAB_STD or rgb_std > MAX_PATCH_RGB_STD:
                continue

            stable.append(
                (
                    lab_std + rgb_std / 10.0,
                    np.median(valid_rgb, axis=0),
                    np.median(valid_lab, axis=0),
                )
            )

    if len(stable) < MIN_STABLE_PATCHES_PER_REGION:
        return np.empty((0, 3)), np.empty((0, 3)), len(stable)

    stable.sort(key=lambda item: item[0])
    selected = stable[:MAX_STABLE_PATCHES_PER_REGION]
    rgb_medians = np.stack([item[1] for item in selected])
    lab_medians = np.stack([item[2] for item in selected])
    return rgb_medians, lab_medians, len(selected)


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

    valid_pixels_rgb, valid_lab = _filter_skin_pixels(pixels)
    valid_count = int(len(valid_pixels_rgb))
    valid_ratio = valid_count / total_count if total_count else 0.0

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

    patch_rgb, patch_lab, stable_patch_count = _stable_patch_medians(image_rgb, mask)
    patch_fallback_used = len(patch_rgb) == 0
    if patch_fallback_used:
        median_rgb = tuple(np.median(valid_pixels_rgb, axis=0).astype(int).tolist())
        median_lab = tuple(np.median(valid_lab, axis=0).tolist())
        if stable_patch_count > 0:
            warnings.append(
                f"{name.replace('_', ' ').title()} had only {stable_patch_count} stable patch(es); "
                "using full-region median fallback."
            )
    else:
        median_rgb = tuple(np.median(patch_rgb, axis=0).astype(int).tolist())
        median_lab = tuple(np.median(patch_lab, axis=0).tolist())

    return RegionSkinResult(
        name=name,
        total_pixel_count=total_count,
        valid_pixel_count=valid_count,
        valid_ratio=valid_ratio,
        median_rgb=median_rgb,
        median_lab=median_lab,
        reliable=valid_count >= MIN_VALID_PIXELS_PER_REGION,
        stable_patch_count=stable_patch_count if not patch_fallback_used else 0,
        patch_fallback_used=patch_fallback_used,
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
      it is specifically darker than the cheeks (possible facial hair or
      chin/neck shadow), since a jawline that is off-color in other ways
      may still carry useful skin-tone signal.

    No-ops (returns no warnings) if no cheek region is reliable, since
    there is then no trustworthy anchor to compare against.
    """
    warnings: list = []
    anchor_lab = _cheek_anchor_lab(reliable_by_name)
    if anchor_lab is None:
        return warnings

    forehead = reliable_by_name.get(FOREHEAD_NAME)
    if forehead is not None:
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
        darkness_gap = anchor_lab[0] - jawline.median_lab[0]
        if darkness_gap > JAWLINE_DARKNESS_L_THRESHOLD:
            jawline.weight_multiplier = JAWLINE_DOWNWEIGHT_FACTOR
            jawline.downweight_reason = (
                f"Jawline is noticeably darker than the cheeks (L difference {darkness_gap:.1f}); "
                "its weight in the final estimate was reduced (possible facial hair or chin shadow)."
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


def _region_base_weight(region_name: str) -> float:
    return 1.0 if region_name in CHEEK_NAMES else OPTIONAL_REGION_BASE_WEIGHT


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
        elif region.patch_fallback_used and region.median_rgb is not None:
            reasons.append(
                f"{name.replace('_', ' ').title()}: used full-region median fallback."
            )
        if region.status_reason:
            reasons.append(f"{name.replace('_', ' ').title()}: {region.status_reason}")

    if skin_result_fields.get("usable_region_count", 0) < 3:
        reasons.append("Fewer than 3 regions were usable, so extraction confidence is reduced slightly.")
    if skin_result_fields.get("cheek_area_balance", 1.0) < CHEEK_AREA_IMBALANCE_WARNING_RATIO:
        reasons.append("Cheek valid-area imbalance reduced confidence slightly.")
    return reasons


def extract_skin_tone(image_rgb: np.ndarray, masks: dict) -> SkinToneResult:
    """Extract a representative skin color from masked face regions.

    For each of forehead/left_cheek/right_cheek/jawline: filters out
    shadow/highlight luminance extremes and extreme-saturation pixels, then
    takes the median RGB/Lab. Cheeks anchor the trust check: forehead is
    excluded outright if it disagrees strongly with the cheeks (likely
    hair/shadow contamination); jawline is down-weighted, not excluded,
    when it is specifically darker than the cheeks (possible facial hair
    or chin shadow). The remaining (non-excluded) reliable regions are
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
    cheek_area_balance, cheek_area_warning = _cheek_area_balance(region_results)
    if cheek_area_warning:
        warnings.append(cheek_area_warning)

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
                warnings=warnings,
                success=False,
            )
        warnings.append(
            "No region met the minimum valid-pixel threshold; using best-available regions "
            "with reduced confidence."
        )
        combination_regions = fallback_regions

    weights = np.array(
        [
            r.valid_pixel_count * r.weight_multiplier * _region_base_weight(r.name)
            for r in combination_regions
        ],
        dtype=np.float64,
    )
    weights = weights / weights.sum()

    rgb_stack = np.array([r.median_rgb for r in combination_regions], dtype=np.float64)
    lab_stack = np.array([r.median_lab for r in combination_regions], dtype=np.float64)

    final_rgb = tuple(np.round(np.average(rgb_stack, axis=0, weights=weights)).astype(int).tolist())
    final_lab = tuple(np.average(lab_stack, axis=0, weights=weights).tolist())

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
    quality_fields = {
        "included_region_names": included_region_names,
        "excluded_region_names": excluded_region_names,
        "usable_region_count": usable_region_count,
        "cheek_area_balance": cheek_area_balance,
    }
    extraction_quality_reasons = _build_extraction_quality_reasons(region_results, quality_fields)

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
        warnings=warnings,
        success=True,
    )
