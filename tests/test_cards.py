"""
tests/test_cards.py — Unit tests for AutoBot Car Recommendation Card View formatting
"""

import pytest
from ui.app import format_car_recommendation_cards


def test_card_conversion_single_car():
    """Verify that a standard vehicle recommendation block is transformed into a car-card."""
    markdown_input = """Here is a great option for you:

### 1. Tata Nexon EV · Compact SUV (₹14.49 L – ₹19.49 L)

* 💰 **Price:** ₹14.49 Lakh – ₹19.49 Lakh (Ex-Showroom)
* ⛽ **Fuel & Engine:** Electric (EV)
* 📊 **Range:** 465 km per charge
* ⚙️ **Transmission:** Automatic
* 🌟 **Key Features:** ADAS, Sunroof, Fast Charging

> 💡 **Best For:** Eco-conscious city commuters seeking low per-km running costs.
"""

    result = format_car_recommendation_cards(markdown_input)

    assert '<div class="car-card">' in result
    assert '<h3 class="car-name">Tata Nexon EV</h3>' in result
    assert '<span class="car-segment-badge">Compact SUV</span>' in result
    assert '<div class="car-price-badge">' in result
    assert "₹14.49 L – ₹19.49 L" in result
    assert '<div class="car-specs-grid">' in result
    assert '<span class="feature-pill">ADAS</span>' in result
    assert '<div class="car-best-for">' in result
    assert "Eco-conscious city commuters" in result


def test_card_conversion_preserves_plain_text():
    """Verify that general automotive text without vehicle blocks remains unmodified."""
    plain_text = "To jump-start your car, connect the red positive clamp first, then the black clamp."
    result = format_car_recommendation_cards(plain_text)
    assert result == plain_text
    assert '<div class="car-card">' not in result


def test_card_idempotency():
    """Verify that already converted HTML cards are not double-wrapped."""
    html_input = '<div class="car-card"><h3 class="car-name">Maruti Swift</h3></div>'
    result = format_car_recommendation_cards(html_input)
    assert result.count('<div class="car-card">') == 1
