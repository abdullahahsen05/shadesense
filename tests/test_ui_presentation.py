from types import SimpleNamespace

from src.ui_presentation import (
    build_region_decision_guidance,
    build_user_guidance,
    guidance_cards_html,
    readiness_display,
    rgb_to_hex,
    shade_strip_html,
)


def test_readiness_copy_is_plain_language():
    assert readiness_display("ready").title == "Ready to compare shades"
    assert "Top 3" in readiness_display("ready").summary
    assert "Retake" in readiness_display("provisional").summary


def test_repeated_specular_messages_collapse_to_one_lighting_card():
    cards = build_user_guidance(
        "caution",
        {
            "skin": [
                "Forehead: possible specular highlight influence detected.",
                "Left cheek: possible specular highlight influence detected.",
                "Right cheek: possible specular highlight influence detected.",
            ]
        },
    )

    lighting_cards = [card for card in cards if card.category == "Lighting"]
    assert len(lighting_cards) == 1
    assert "handled automatically" in lighting_cards[0].title


def test_provisional_guidance_is_actionable_and_capped():
    cards = build_user_guidance(
        "provisional",
        {
            "capture": ["Image is underexposed and face pose is angled."],
            "mask": ["Glasses frame reflection detected."],
            "skin": ["Independent region support is limited."],
        },
        max_cards=3,
    )

    assert len(cards) == 3
    assert cards[0].title == "Retake one photo for a stronger match"
    assert any(card.category == "Lighting" for card in cards)


def test_color_helpers_validate_and_escape_display_content():
    assert rgb_to_hex((201, 145, 123)) == "#C9917B"
    html = shade_strip_html(
        [{"label": "Choice 1", "name": "<script>", "hex": "#C9917B"}]
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "background:#C9917B" in html


def test_guidance_html_uses_compact_grid():
    cards = build_user_guidance("ready", {})
    html = guidance_cards_html(cards)
    assert 'class="ss-guidance-grid"' in html
    assert "No major capture issues detected" in html


def test_region_decisions_name_exclusions_and_reduced_weights():
    cards = build_region_decision_guidance(
        {
            "forehead": SimpleNamespace(
                excluded=True,
                exclusion_reason="Hairline contamination was detected.",
                weight_multiplier=0.0,
            ),
            "jawline": SimpleNamespace(
                excluded=False,
                downweight_reason="It disagreed with both cheeks.",
                weight_multiplier=0.12,
            ),
            "left_cheek": SimpleNamespace(
                excluded=False,
                downweight_reason=None,
                weight_multiplier=1.0,
            ),
        }
    )

    assert len(cards) == 2
    assert cards[0].title == "One facial region was excluded"
    assert "Forehead: Hairline contamination" in cards[0].message
    assert cards[1].title == "One region had reduced influence"
    assert "Jawline retained 12% influence" in cards[1].message
    assert "disagreed with both cheeks" in cards[1].message
