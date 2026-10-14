"""
tests/test_emi.py — Unit tests for AutoBot EMI calculation tool
"""

import pytest
from tools.car_tools import calculate_emi


def test_calculate_emi_standard():
    """Verify EMI for a standard 10 Lakh loan at 9% for 5 years."""
    result = calculate_emi(principal=1_000_000, annual_rate=9.0, tenure_months=60)
    assert "monthly_emi" in result
    assert "monthly_emi_raw" in result
    assert "total_interest" in result
    assert "total_payment" in result
    # Monthly EMI raw for 10L @ 9% for 5yr is approx 20,758
    assert result["monthly_emi_raw"] > 20000
    assert result["monthly_emi_raw"] < 21500
    assert "₹" in result["monthly_emi"]


def test_calculate_emi_zero_interest():
    """Verify EMI when interest rate is 0% (straight division)."""
    result = calculate_emi(principal=600_000, annual_rate=0.0, tenure_months=60)
    assert result["monthly_emi_raw"] == 10_000.0
    assert result["total_interest"] == "₹0"
    assert "₹6.00L" in result["total_payment"] or "₹600,000" in result["total_payment"]


def test_calculate_emi_formatting():
    """Verify that formatted rupee currency strings are present in output."""
    result = calculate_emi(principal=500_000, annual_rate=10.0, tenure_months=36)
    assert "monthly_emi" in result
    assert "₹" in result["monthly_emi"]
    assert "total_interest" in result
    assert "total_payment" in result
