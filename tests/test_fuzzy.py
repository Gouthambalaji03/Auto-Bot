"""
tests/test_fuzzy.py — Unit tests for AutoBot RapidFuzz search engine & synonyms
"""

import pytest
from db.fuzzy_queries import (
    fuzzy_engine,
    FUEL_ALIASES,
    SEGMENT_SYNONYMS,
    BRAND_ALIASES,
)


def test_fuel_aliases():
    """Verify alias mapping for various natural language fuel terms."""
    assert FUEL_ALIASES["ev"] == "EV"
    assert FUEL_ALIASES["electric"] == "EV"
    assert FUEL_ALIASES["bev"] == "EV"
    assert FUEL_ALIASES["petrol"] == "Petrol"
    assert FUEL_ALIASES["gasoline"] == "Petrol"
    assert FUEL_ALIASES["diesel"] == "Diesel"
    assert FUEL_ALIASES["cng"] == "CNG"
    assert FUEL_ALIASES["hybrid"] == "Hybrid"


def test_segment_synonyms():
    """Verify segment synonym groupings."""
    assert "suv" in SEGMENT_SYNONYMS["suv"]
    assert "crossover" in SEGMENT_SYNONYMS["suv"]
    assert "hatchback" in SEGMENT_SYNONYMS["city car"]
    assert "mpv" in SEGMENT_SYNONYMS["7 seater"]


def test_brand_aliases():
    """Verify brand alias normalizations."""
    assert BRAND_ALIASES["maruti"] == "Maruti Suzuki"
    assert BRAND_ALIASES["suzuki"] == "Maruti Suzuki"
    assert BRAND_ALIASES["tata motors"] == "Tata Motors"
    assert BRAND_ALIASES["vw"] == "Volkswagen"


def test_fuzzy_issue_search():
    """Verify fuzzy issue matching on sample issues."""
    sample_issues = {
        "vibration_at_high_speed": {
            "causes": ["Wheel unbalance", "Warped brake rotor"],
            "most_likely": "Wheel imbalance",
            "severity": "Medium",
            "cost_range": "₹500 – ₹2,000",
        },
        "engine_overheating": {
            "causes": ["Coolant leak", "Failed thermostat"],
            "most_likely": "Low coolant level",
            "severity": "High",
            "cost_range": "₹1,000 – ₹8,000",
        },
    }

    match = fuzzy_engine.search_issue("car vibration at high speed", sample_issues)
    assert match is not None
    assert match["most_likely"] == "Wheel imbalance"

    overheat_match = fuzzy_engine.search_issue("car engine overheating hot coolant", sample_issues)
    assert overheat_match is not None
    assert overheat_match["severity"] == "High"
