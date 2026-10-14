"""Domain schemas retained for optional fuzzy catalogue search.

The active runtime does not use these classes for intent routing or final LLM
output. Intent classification and safe tool selection happen inside the single
Pydantic AI automotive agent.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


IntentType = Literal["recommend", "diagnostic", "service", "emi", "general"]
SortByOption = Literal["price_asc", "price_desc", "mileage_desc", "seating_desc", "relevance"]


class FuzzyCarFilter(BaseModel):
    """Optional, typed inputs for the local fuzzy catalogue utility."""

    budget_lakh_min: Optional[float] = None
    budget_lakh_max: Optional[float] = None
    budget_tolerance_pct: float = Field(default=0.20, ge=0.0, le=1.0)
    fuel_type: Optional[str] = None
    segment: Optional[str] = None
    min_seating: Optional[int] = None
    transmission: Optional[str] = None
    car_name_query: Optional[str] = None
    feature_keywords: list[str] = Field(default_factory=list)
    issue_keyword: Optional[str] = None
    brand: Optional[str] = None


class QueryPlan(BaseModel):
    """Typed plan consumed only by ``db.fuzzy_queries.FuzzySearchEngine``."""

    intent: IntentType
    filters: FuzzyCarFilter = Field(default_factory=FuzzyCarFilter)
    sort_by: SortByOption = "relevance"
    max_results: int = Field(default=5, ge=1, le=20)
    reasoning: str = ""
    emi_price_lakh: Optional[float] = None
    emi_interest_rate: Optional[float] = None
    emi_tenure_years: Optional[int] = None
