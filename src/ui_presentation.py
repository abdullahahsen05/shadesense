"""User-facing presentation helpers for the Streamlit results experience."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable, Mapping, Sequence


APP_STYLES = """
<style>
:root {
    --ss-ink: #0c1017;
    --ss-slate: #171d27;
    --ss-slate-soft: #202836;
    --ss-line: #313b4b;
    --ss-ivory: #f3eee9;
    --ss-muted: #aeb7c5;
    --ss-complexion: #c9917b;
    --ss-ready: #59b8a6;
    --ss-caution: #d6a64a;
    --ss-provisional: #d46a68;
}

.stApp {
    background:
        radial-gradient(circle at 92% 3%, rgba(201, 145, 123, 0.10), transparent 24rem),
        var(--ss-ink);
}

.block-container {
    max-width: 1240px;
    padding-top: 4rem;
    padding-bottom: 5rem;
}

.ss-kicker {
    color: var(--ss-complexion);
    font: 700 0.73rem/1.2 ui-monospace, "Cascadia Code", monospace;
    letter-spacing: 0.15em;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
}

.ss-title {
    color: var(--ss-ivory);
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: clamp(2.3rem, 5vw, 4.5rem);
    font-weight: 500 !important;
    letter-spacing: -0.045em;
    line-height: 0.98;
    margin: 0;
}

.ss-lede {
    color: var(--ss-muted);
    font-size: 1.02rem;
    line-height: 1.65;
    margin: 1rem 0 1.5rem;
    max-width: 48rem;
}

.ss-result-hero {
    background: linear-gradient(135deg, rgba(32, 40, 54, 0.98), rgba(19, 25, 35, 0.98));
    border: 1px solid var(--ss-line);
    border-left: 5px solid var(--ss-accent, var(--ss-caution));
    border-radius: 18px;
    margin: 0.75rem 0 1.4rem;
    padding: 1.25rem 1.4rem 1.3rem;
}

.ss-result-hero.ready { --ss-accent: var(--ss-ready); }
.ss-result-hero.caution { --ss-accent: var(--ss-caution); }
.ss-result-hero.provisional,
.ss-result-hero.unavailable { --ss-accent: var(--ss-provisional); }

.ss-result-row {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    justify-content: space-between;
}

.ss-status {
    background: color-mix(in srgb, var(--ss-accent) 16%, transparent);
    border: 1px solid color-mix(in srgb, var(--ss-accent) 55%, transparent);
    border-radius: 999px;
    color: var(--ss-accent);
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    padding: 0.35rem 0.7rem;
    text-transform: uppercase;
}

.ss-score {
    color: var(--ss-ivory);
    font: 650 0.85rem/1.2 ui-monospace, "Cascadia Code", monospace;
}

.ss-result-hero h2 {
    color: var(--ss-ivory);
    font-size: clamp(1.45rem, 3vw, 2rem);
    line-height: 1.15;
    margin: 0.9rem 0 0.45rem;
}

.ss-result-hero p {
    color: var(--ss-muted);
    line-height: 1.55;
    margin: 0;
    max-width: 58rem;
}

.ss-guidance-grid {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    margin: 0.55rem 0 1.4rem;
}

.ss-guidance {
    background: rgba(23, 29, 39, 0.78);
    border: 1px solid var(--ss-line);
    border-radius: 14px;
    min-height: 7.2rem;
    padding: 1rem;
}

.ss-guidance.info { border-top: 3px solid #7196c4; }
.ss-guidance.caution { border-top: 3px solid var(--ss-caution); }
.ss-guidance.good { border-top: 3px solid var(--ss-ready); }

.ss-guidance-label {
    color: var(--ss-muted);
    font: 700 0.68rem/1.2 ui-monospace, "Cascadia Code", monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.ss-guidance strong {
    color: var(--ss-ivory);
    display: block;
    font-size: 0.98rem;
    margin: 0.45rem 0 0.35rem;
}

.ss-guidance p {
    color: var(--ss-muted);
    font-size: 0.88rem;
    line-height: 1.45;
    margin: 0;
}

.ss-shade-strip {
    background: rgba(23, 29, 39, 0.72);
    border: 1px solid var(--ss-line);
    border-radius: 16px;
    display: grid;
    gap: 0.8rem;
    grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
    margin: 0.8rem 0 1.45rem;
    padding: 0.9rem;
}

.ss-shade-stop {
    min-width: 0;
}

.ss-shade-color {
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 11px;
    box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.08);
    height: 62px;
    margin-bottom: 0.55rem;
}

.ss-shade-label {
    color: var(--ss-muted);
    font: 700 0.66rem/1.2 ui-monospace, "Cascadia Code", monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.ss-shade-name {
    color: var(--ss-ivory);
    font-size: 0.86rem;
    margin-top: 0.25rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(23, 29, 39, 0.72);
    border-color: var(--ss-line);
    border-radius: 16px;
}

div[data-testid="stMetric"] {
    background: rgba(23, 29, 39, 0.66);
    border: 1px solid var(--ss-line);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    background: rgba(23, 29, 39, 0.54);
    border-color: var(--ss-line);
    border-radius: 12px;
}

.ss-section-note {
    color: var(--ss-muted);
    margin: -0.3rem 0 1rem;
}

@media (max-width: 760px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 3.5rem;
    }
    .ss-title { font-size: 2.55rem; }
    .ss-result-hero { padding: 1rem; }
    .ss-shade-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


@dataclass(frozen=True)
class GuidanceCard:
    """One concise, user-facing capture or analysis note."""

    category: str
    title: str
    message: str
    tone: str = "info"


@dataclass(frozen=True)
class ReadinessDisplay:
    """Plain-language readiness content for the primary result card."""

    state: str
    title: str
    summary: str


def readiness_display(state: str) -> ReadinessDisplay:
    """Translate an internal readiness state into clear user language."""
    normalized = (state or "provisional").strip().lower()
    if normalized == "ready":
        return ReadinessDisplay(
            state="ready",
            title="Ready to compare shades",
            summary=(
                "The photo and extracted tone are stable enough for catalog "
                "comparison. Use the Top 3 as a practical shortlist."
            ),
        )
    if normalized == "caution":
        return ReadinessDisplay(
            state="caution",
            title="Usable, with some caution",
            summary=(
                "The result is usable, but lighting or regional agreement may "
                "move the exact product. Compare the Top 3 before choosing."
            ),
        )
    if normalized == "unavailable":
        return ReadinessDisplay(
            state="unavailable",
            title="A clearer photo is needed",
            summary=(
                "ShadeSense could not collect enough reliable facial evidence. "
                "Retake the photo in soft, even light with the face fully visible."
            ),
        )
    return ReadinessDisplay(
        state="provisional",
        title="Treat this result as provisional",
        summary=(
            "The estimated color family may be useful, but this capture is not "
            "strong enough for a confident exact-product choice. Retake it in "
            "soft, even daylight."
        ),
    )


def _contains(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _flatten_warning_groups(
    warning_groups: Mapping[str, Iterable[str]],
) -> tuple[str, dict[str, str]]:
    source_text: dict[str, str] = {}
    all_messages: list[str] = []
    for source, messages in warning_groups.items():
        deduplicated = list(dict.fromkeys(str(message).strip() for message in messages if message))
        source_text[source] = " ".join(deduplicated).casefold()
        all_messages.extend(deduplicated)
    return " ".join(all_messages).casefold(), source_text


def build_user_guidance(
    readiness_state: str,
    warning_groups: Mapping[str, Iterable[str]],
    *,
    max_cards: int = 3,
) -> list[GuidanceCard]:
    """Collapse technical warnings into a few actionable user messages."""
    all_text, source_text = _flatten_warning_groups(warning_groups)
    cards: list[GuidanceCard] = []
    state = (readiness_state or "provisional").casefold()

    eyewear = _contains(
        all_text,
        ("eyewear", "glasses", "spectacle", "lens reflection", "frame reflection"),
    )
    low_light = _contains(
        all_text,
        ("underexpos", "too dark", "low light", "low-signal", "low signal"),
    )
    bright_light = _contains(
        all_text,
        ("overexpos", "too bright", "clipp", "washed out"),
    )
    uneven_light = _contains(
        all_text,
        ("uneven", "asymmetr", "one side", "left/right", "side lighting"),
    )
    shine = _contains(
        all_text,
        ("specular", "highlight influence", "facial shine", "gloss"),
    )
    color_cast = _contains(all_text, ("color cast", "colour cast", "white balance"))
    pose = _contains(
        all_text,
        ("pose", "angled", "angle", "foreshorten", "turned", "face camera"),
    )
    region_limited = _contains(
        " ".join(
            source_text.get(source, "")
            for source in ("skin", "extraction", "mask", "readiness")
        ),
        (
            "limited",
            "single region",
            "dominant region",
            "jawline differs",
            "reduced",
            "contamin",
            "disagree",
            "insufficient region",
        ),
    )

    if state in {"provisional", "unavailable"}:
        cards.append(
            GuidanceCard(
                category="Next step",
                title="Retake one photo for a stronger match",
                message=(
                    "Face the camera directly in soft daylight, avoid filters, "
                    "and keep both cheeks and the side jaw visible."
                ),
                tone="caution",
            )
        )

    if low_light or bright_light or uneven_light or color_cast:
        details: list[str] = []
        if low_light:
            details.append("the face is too dim")
        if bright_light:
            details.append("some skin areas are too bright")
        if uneven_light:
            details.append("light differs across the face")
        if color_cast:
            details.append("the image has a color cast")
        cards.append(
            GuidanceCard(
                category="Lighting",
                title="Lighting may shift the exact shade",
                message=(
                    f"ShadeSense detected {', '.join(details)}. "
                    "Move into soft, even daylight for a more stable comparison."
                ),
                tone="caution",
            )
        )
    elif shine:
        cards.append(
            GuidanceCard(
                category="Lighting",
                title="Facial shine was handled automatically",
                message=(
                    "Shiny patches were given less influence so they do not make "
                    "the foundation target appear too light."
                ),
                tone="info",
            )
        )

    if eyewear:
        cards.append(
            GuidanceCard(
                category="Obstruction",
                title="Glasses or reflections were excluded",
                message=(
                    "Pixels near the frames and lenses were removed from skin "
                    "sampling. Removing glasses can still improve consistency."
                ),
                tone="info",
            )
        )

    if pose:
        cards.append(
            GuidanceCard(
                category="Position",
                title="A straighter pose would improve balance",
                message=(
                    "One side of the face had less reliable evidence. Look "
                    "straight at the camera with the full jaw visible."
                ),
                tone="caution",
            )
        )

    if region_limited:
        cards.append(
            GuidanceCard(
                category="Skin evidence",
                title="The strongest skin regions carried more weight",
                message=(
                    "Unreliable forehead, cheek, or jaw patches were reduced. "
                    "The result relies on the cleanest retained facial evidence."
                ),
                tone="info",
            )
        )

    if not cards:
        cards.append(
            GuidanceCard(
                category="Photo quality",
                title="No major capture issues detected",
                message=(
                    "The visible facial regions provide consistent evidence for "
                    "shade comparison."
                ),
                tone="good",
            )
        )

    return cards[: max(1, max_cards)]


def build_region_decision_guidance(
    region_results: Mapping[str, object],
) -> list[GuidanceCard]:
    """Explain region exclusion and weighting decisions in user-facing language."""
    excluded_details: list[str] = []
    reduced_details: list[str] = []

    for region_name, region in region_results.items():
        label = region_name.replace("_", " ").title()
        excluded = bool(getattr(region, "excluded", False))
        weight = float(getattr(region, "weight_multiplier", 1.0))
        if excluded:
            reason = (
                getattr(region, "exclusion_reason", None)
                or getattr(region, "status_reason", None)
                or "it did not provide reliable skin evidence"
            )
            excluded_details.append(f"{label}: {str(reason).strip()}")
        elif weight < 0.995:
            reason = (
                getattr(region, "downweight_reason", None)
                or getattr(region, "status_reason", None)
                or "its evidence was less reliable than the other regions"
            )
            reduced_details.append(
                f"{label} retained {max(0.0, weight):.0%} influence: "
                f"{str(reason).strip()}"
            )

    cards: list[GuidanceCard] = []
    if excluded_details:
        count = len(excluded_details)
        cards.append(
            GuidanceCard(
                category="Region decision",
                title=(
                    "One facial region was excluded"
                    if count == 1
                    else f"{count} facial regions were excluded"
                ),
                message=" ".join(excluded_details),
                tone="caution",
            )
        )
    if reduced_details:
        count = len(reduced_details)
        cards.append(
            GuidanceCard(
                category="Region decision",
                title=(
                    "One region had reduced influence"
                    if count == 1
                    else f"{count} regions had reduced influence"
                ),
                message=" ".join(reduced_details),
                tone="info",
            )
        )
    return cards


def guidance_cards_html(cards: Sequence[GuidanceCard]) -> str:
    """Render guidance cards as a compact responsive grid."""
    items = []
    for card in cards:
        tone = card.tone if card.tone in {"info", "caution", "good"} else "info"
        items.append(
            '<div class="ss-guidance '
            + tone
            + '"><div class="ss-guidance-label">'
            + escape(card.category)
            + "</div><strong>"
            + escape(card.title)
            + "</strong><p>"
            + escape(card.message)
            + "</p></div>"
        )
    return '<div class="ss-guidance-grid">' + "".join(items) + "</div>"


def rgb_to_hex(rgb: Sequence[int]) -> str:
    """Return a validated display HEX value from an RGB triplet."""
    values = [max(0, min(255, int(value))) for value in rgb[:3]]
    if len(values) != 3:
        raise ValueError("RGB must contain exactly three channels.")
    return "#" + "".join(f"{value:02X}" for value in values)


def shade_strip_html(stops: Sequence[Mapping[str, str]]) -> str:
    """Render the measured-to-recommended color evidence strip."""
    rendered = []
    for stop in stops:
        color = escape(str(stop["hex"]))
        rendered.append(
            '<div class="ss-shade-stop">'
            f'<div class="ss-shade-color" style="background:{color}"></div>'
            '<div class="ss-shade-label">'
            + escape(str(stop["label"]))
            + "</div><div class=\"ss-shade-name\">"
            + escape(str(stop["name"]))
            + "</div></div>"
        )
    return '<div class="ss-shade-strip">' + "".join(rendered) + "</div>"
