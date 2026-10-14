"""
AutoBot Tools — car data retrieval and calculations

Data source: PostgreSQL (via db/queries.py) with TTL in-memory cache.
"""

from typing import Optional
from db.queries import (
    db_get_all_cars,
    db_filter_by_budget,
    db_filter_by_fuel,
    db_filter_by_segment,
    db_get_car_by_name,
    db_get_common_issue,
    db_get_service_intervals,
)


# ─────────────────────────────────────────────
# Tool: Get All Cars
# ─────────────────────────────────────────────
def get_all_cars() -> list[dict]:
    """Returns all cars from PostgreSQL database."""
    return db_get_all_cars()


# ─────────────────────────────────────────────
# Tool: Filter Cars by Budget
# ─────────────────────────────────────────────
def filter_cars_by_budget(max_budget_lakh: float) -> list[dict]:
    """Filter cars within a given budget in lakhs."""
    return db_filter_by_budget(max_budget_lakh)


# ─────────────────────────────────────────────
# Tool: Filter Cars by Fuel Type
# ─────────────────────────────────────────────
def filter_cars_by_fuel(fuel_type: str) -> list[dict]:
    """Filter cars by fuel type (Petrol, Diesel, EV, CNG, Hybrid)."""
    return db_filter_by_fuel(fuel_type)


# ─────────────────────────────────────────────
# Tool: Filter Cars by Segment
# ─────────────────────────────────────────────
def filter_cars_by_segment(segment: str) -> list[dict]:
    """Filter cars by segment (SUV, Sedan, Hatchback, MUV, etc.)."""
    return db_filter_by_segment(segment)


# ─────────────────────────────────────────────
# Tool: Get Car by Name
# ─────────────────────────────────────────────
def get_car_by_name(name: str) -> Optional[dict]:
    """Find a specific car by name (partial match)."""
    return db_get_car_by_name(name)


# ─────────────────────────────────────────────
# Tool: Get Common Issue Info
# ─────────────────────────────────────────────
def get_common_issue_info(issue_keyword: str) -> Optional[dict]:
    """Get info about a common car issue by keyword."""
    return db_get_common_issue(issue_keyword)


# ─────────────────────────────────────────────
# Tool: Get Service Intervals
# ─────────────────────────────────────────────
def get_service_intervals() -> dict:
    """Returns standard service intervals and costs."""
    return db_get_service_intervals()


# ─────────────────────────────────────────────
# Tool: Calculate EMI (pure math — no DB)
# ─────────────────────────────────────────────
def calculate_emi(
    principal: float,
    annual_rate: float,
    tenure_months: int
) -> dict:
    """
    Calculate car loan EMI.

    Args:
        principal:      Loan amount in INR
        annual_rate:    Annual interest rate in %
        tenure_months:  Loan tenure in months

    Returns:
        Dict with monthly_emi, total_interest, total_payment, principal (formatted strings)
    """
    monthly_rate = annual_rate / (12 * 100)

    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months
        emi = emi / ((1 + monthly_rate) ** tenure_months - 1)

    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    def fmt(val: float) -> str:
        if val >= 100000:
            return f"₹{val/100000:.2f}L"
        return f"₹{val:,.0f}"

    return {
        "monthly_emi":      fmt(emi),
        "monthly_emi_raw":  round(emi, 2),
        "total_interest":   fmt(total_interest),
        "total_payment":    fmt(total_payment),
        "principal":        fmt(principal),
    }
