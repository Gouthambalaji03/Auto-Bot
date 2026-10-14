"""
db/fuzzy_queries.py — Fuzzy Search Engine for AutoBot

Provides a single class `FuzzySearchEngine` with three public methods,
one per intent type:

    fuzzy_engine.search_cars(plan, cars)         → recommend intent
    fuzzy_engine.search_issue(keyword, issues)   → diagnostic intent
    fuzzy_engine.search_service(keyword, ivs)    → service intent

Each method follows the same contract:
    normalise → score → sort → return

Usage (anywhere in the codebase):
    from db.fuzzy_queries import fuzzy_engine

    # Recommend
    results = fuzzy_engine.search_cars(query_plan, all_cars)

    # Diagnostic
    issue = fuzzy_engine.search_issue("car shaking badly", all_issues)

    # Service
    relevant = fuzzy_engine.search_service("timing belt nexon", intervals)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from rapidfuzz import fuzz, process as rfuzz_process

if TYPE_CHECKING:
    from models.schemas import QueryPlan


# ─────────────────────────────────────────────
# Alias / Synonym Maps
# User input → canonical DB value
# ─────────────────────────────────────────────

FUEL_ALIASES: dict[str, str] = {
    # Electric
    "ev": "EV", "electric": "EV", "electrical": "EV", "battery": "EV",
    "bev": "EV", "electric vehicle": "EV",
    # Petrol
    "petrol": "Petrol", "gasoline": "Petrol", "gas": "Petrol", "patrol": "Petrol",
    # Diesel
    "diesel": "Diesel", "disel": "Diesel", "desel": "Diesel",
    # CNG
    "cng": "CNG", "compressed natural gas": "CNG", "natural gas": "CNG",
    # Hybrid
    "hybrid": "Hybrid", "mild hybrid": "Hybrid", "strong hybrid": "Hybrid",
    "self charging": "Hybrid", "self-charging": "Hybrid",
}

SEGMENT_SYNONYMS: dict[str, list[str]] = {
    # User term → list of DB segment keywords to match (case-insensitive)
    "suv":         ["suv", "crossover"],
    "sedan":       ["sedan"],
    "hatchback":   ["hatchback", "hatch"],
    "mpv":         ["mpv", "muv", "van"],
    "muv":         ["mpv", "muv"],
    "compact suv": ["suv", "crossover"],
    "family car":  ["suv", "mpv", "muv"],
    "city car":    ["hatchback", "sedan"],
    "highway car": ["sedan", "suv"],
    "off road":    ["suv"],
    "7 seater":    ["suv", "mpv", "muv"],
    "8 seater":    ["mpv", "muv"],
    "small car":   ["hatchback"],
    "big car":     ["suv", "mpv"],
}

TRANSMISSION_ALIASES: dict[str, str] = {
    "auto": "Automatic", "automatic": "Automatic", "at": "Automatic",
    "amt": "Automatic",   # AMT = robotised manual, treated as auto in Indian market
    "manual": "Manual", "mt": "Manual", "stick": "Manual",
    "cvt": "CVT", "dct": "DCT", "dual clutch": "DCT",
}

BRAND_ALIASES: dict[str, str] = {
    "maruti": "Maruti Suzuki", "suzuki": "Maruti Suzuki", "ms": "Maruti Suzuki",
    "hyundai": "Hyundai", "hundai": "Hyundai", "hundayai": "Hyundai",
    "tata": "Tata Motors", "tata motors": "Tata Motors",
    "mahindra": "Mahindra", "m&m": "Mahindra",
    "honda": "Honda",
    "toyota": "Toyota",
    "kia": "Kia",
    "renault": "Renault",
    "volkswagen": "Volkswagen", "vw": "Volkswagen",
    "skoda": "Skoda",
    "mg": "MG", "morris garages": "MG",
    "jeep": "Jeep",
    "ford": "Ford",
}


# ─────────────────────────────────────────────
# Score Weights
# All scoring constants in ONE place — easy to tune
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class _Weights:
    budget_exact:     float = 30.0   # price_min ≤ user budget
    budget_stretch:   float = 10.0   # price_min ≤ budget × (1 + tolerance)
    budget_below_min: float = -10.0  # price_max < user min budget
    fuel_exact:       float = 25.0   # exact fuel match
    fuel_fuzzy:       float = 15.0   # fuzzy fuel match (score ≥ 70)
    fuel_mismatch:    float = -15.0  # fuel specified but car doesn't have it
    segment_exact:    float = 20.0   # segment keyword found in car.segment
    segment_fuzzy:    float = 10.0   # fuzzy segment match (score ≥ 65)
    seating_ok:       float = 10.0   # car.seating ≥ min_seating
    seating_fail:     float = -20.0  # car.seating < min_seating
    brand_exact:      float = 15.0   # brand fuzzy score ≥ 80
    brand_partial:    float =  7.0   # brand fuzzy score ≥ 60
    transmission:     float = 10.0   # transmission match
    feature_per_hit:  float =  5.0   # per feature keyword hit
    feature_max:      float = 15.0   # cap on feature score
    popularity_per:   float =  1.5   # per popular_in entry
    popularity_max:   float =  5.0   # cap on popularity bonus

W = _Weights()   # singleton instance used throughout


# ─────────────────────────────────────────────
# Fuzzy Search Engine
# ─────────────────────────────────────────────

class FuzzySearchEngine:
    """
    Unified fuzzy search engine for AutoBot.

    Instantiated once at module level as `fuzzy_engine`.
    All methods are stateless — safe for concurrent async use.
    """

    # ── Public API ────────────────────────────────────────────────

    def search_cars(self, plan: "QueryPlan", all_cars: list[dict]) -> list[dict]:
        """
        Score and rank all cars against the QueryPlan.

        Args:
            plan:     QueryPlan from query_agent (contains filters + sort strategy)
            all_cars: Full car list from get_all_cars()

        Returns:
            Ranked list of car dicts (up to plan.max_results),
            each with a '_relevance_score' key added.

        Filtering stages (applied in order):
            1. Budget hard gate + soft score
            2. Fuel type normalisation + score
            3. Segment synonym expansion + score
            4. Seating requirement
            5. Brand fuzzy match
            6. Transmission preference
            7. Feature keyword overlap
            8. Popularity bonus
        """
        f = plan.filters

        # Normalise all filter inputs once, up front
        fuel  = self._normalise(f.fuel_type,    FUEL_ALIASES,    threshold=70)
        tx    = self._normalise(f.transmission, TRANSMISSION_ALIASES, threshold=70)
        brand = self._normalise(f.brand,        BRAND_ALIASES,   threshold=65)
        seg_keywords = self._expand_segment(f.segment)

        print(f"[FUZZY CARS] budget=Rs.{f.budget_lakh_max}L | fuel={fuel} | "
              f"segment={seg_keywords} | brand={brand} | tx={tx} | "
              f"features={f.feature_keywords} | sort={plan.sort_by}")

        # Short-circuit: if user named a specific car, do a direct name lookup
        if f.car_name_query:
            return self._lookup_by_name(f.car_name_query, all_cars, plan.max_results)

        # Score every car
        scored: list[tuple[dict, float]] = []
        for car in all_cars:
            score = self._score_car(car, f, fuel, tx, brand, seg_keywords)
            if score is not None:          # None = hard-excluded by budget gate
                scored.append((car, max(score, 0.0)))
                if score > 0:
                    print(f"   + {car['name']:22s}  score={score:5.1f}")

        # Sort by user's chosen strategy
        sorted_cars = self._apply_sort(scored, plan.sort_by)

        # Attach score and trim to max_results
        results = [
            {**car, "_relevance_score": round(sc, 1)}
            for car, sc in sorted_cars[:plan.max_results]
        ]
        print(f"[FUZZY CARS] returning {len(results)} / {len(scored)} candidates")
        return results

    def search_issue(self, keyword: str, all_issues: dict) -> Optional[dict]:
        """
        Fuzzy-match a user symptom against the common_issues keys.

        Safety note: threshold is 65 (not 45) because diagnostic data is
        safety-critical — a wrong match could give wrong severity or repair cost.

        Args:
            keyword:    Raw symptom string, e.g. "car shakes at high speed"
            all_issues: Dict of {issue_key: issue_data} from DB or JSON

        Returns:
            Best matching issue dict, or None if no match above threshold.
        """
        if not all_issues or not keyword:
            return None

        issue_keys    = list(all_issues.keys())
        display_keys  = [k.replace("_", " ") for k in issue_keys]

        result = rfuzz_process.extractOne(
            keyword.lower(),
            display_keys,
            scorer=fuzz.token_set_ratio,
            score_cutoff=65.0,          # raised from 45 → precision over recall
        )

        if result:
            matched_display, score, idx = result
            matched_key = issue_keys[idx]
            print(f"[FUZZY ISSUE] '{keyword}' -> '{matched_key}' (score={score:.0f})")
            return all_issues[matched_key]

        print(f"[FUZZY ISSUE] '{keyword}' — no match above threshold (65)")
        return None

    def search_service(self, keyword: str, intervals: dict) -> dict:
        """
        Fuzzy-match a user service query against interval item keys.
        Returns only the RELEVANT intervals (not all 10 dumped into the prompt).

        Args:
            keyword:   User query string, e.g. "timing belt replacement nexon"
            intervals: Full dict of {item_key: interval_data}

        Returns:
            Dict of {item_key: interval_data} for top matching items.
            Falls back to the 5 most common items if nothing matches well.
        """
        if not intervals or not keyword:
            return self._default_service_items(intervals)

        display_keys = [k.replace("_", " ") for k in intervals]

        # Extract top-3 matches above threshold
        results = rfuzz_process.extract(
            keyword.lower(),
            display_keys,
            scorer=fuzz.token_set_ratio,
            score_cutoff=50.0,
            limit=3,
        )

        if results:
            matched: dict = {}
            for _display, score, idx in results:
                key = list(intervals.keys())[idx]
                matched[key] = intervals[key]
                print(f"[FUZZY SERVICE] '{keyword}' -> '{key}' (score={score:.0f})")
            return matched

        print(f"[FUZZY SERVICE] '{keyword}' — no match, returning defaults")
        return self._default_service_items(intervals)

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _normalise(
        raw: Optional[str],
        alias_map: dict,
        threshold: float = 70.0,
    ) -> Optional[str]:
        """
        Maps a raw user string → canonical DB value via alias map.
        Falls back to fuzzy match on alias keys if exact lookup fails.
        Returns None if raw is falsy.
        """
        if not raw:
            return None
        key = raw.lower().strip()
        # Exact alias lookup
        if key in alias_map:
            return alias_map[key]
        # Fuzzy alias lookup
        result = rfuzz_process.extractOne(
            key,
            list(alias_map.keys()),
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
        )
        return alias_map[result[0]] if result else raw.title()

    @staticmethod
    def _expand_segment(raw: Optional[str]) -> list[str]:
        """
        Maps a raw segment string → list of DB segment keywords to match.
        E.g. "family car" → ["suv", "mpv", "muv"]
        """
        if not raw:
            return []
        key = raw.lower().strip()
        if key in SEGMENT_SYNONYMS:
            return SEGMENT_SYNONYMS[key]
        # Fuzzy match the synonym map keys
        result = rfuzz_process.extractOne(
            key,
            list(SEGMENT_SYNONYMS.keys()),
            scorer=fuzz.token_set_ratio,
            score_cutoff=65,
        )
        return SEGMENT_SYNONYMS[result[0]] if result else [key]

    def _score_car(
        self,
        car: dict,
        f,           # FuzzyCarFilter
        fuel: Optional[str],
        tx: Optional[str],
        brand: Optional[str],
        seg_keywords: list[str],
    ) -> Optional[float]:
        """
        Compute composite relevance score for one car.
        Returns None if the car is hard-excluded (e.g. far over budget).
        """
        score = 0.0
        p_min = car["price_lakh"]["min"]
        p_max = car["price_lakh"]["max"]

        # ── Stage 1: Budget ───────────────────────────────────────
        if f.budget_lakh_max is not None:
            hard_max = f.budget_lakh_max * (1 + f.budget_tolerance_pct)
            if p_min > hard_max:
                return None                  # hard exclude — too expensive
            score += W.budget_exact if p_min <= f.budget_lakh_max else W.budget_stretch

        if f.budget_lakh_min is not None and p_max < f.budget_lakh_min:
            score += W.budget_below_min

        # ── Stage 2: Fuel ─────────────────────────────────────────
        if fuel:
            car_fuels = [x.lower() for x in car.get("fuel_type", [])]
            if fuel.lower() in car_fuels:
                score += W.fuel_exact
            else:
                best_fuel_score = max(
                    (fuzz.token_set_ratio(fuel, f_val) for f_val in car.get("fuel_type", [""])),
                    default=0.0,
                )
                if best_fuel_score >= 70:
                    score += W.fuel_fuzzy
                else:
                    score += W.fuel_mismatch

        # ── Stage 3: Segment ──────────────────────────────────────
        if seg_keywords:
            car_seg = car.get("segment", "").lower()
            if any(kw in car_seg for kw in seg_keywords):
                score += W.segment_exact
            else:
                best_seg_score = max(
                    (fuzz.token_set_ratio(kw, car_seg) for kw in seg_keywords),
                    default=0.0,
                )
                if best_seg_score >= 65:
                    score += W.segment_fuzzy

        # ── Stage 4: Seating ──────────────────────────────────────
        if f.min_seating is not None:
            car_seats = car.get("seating", 5)
            score += W.seating_ok if car_seats >= f.min_seating else W.seating_fail

        # ── Stage 5: Brand ────────────────────────────────────────
        if brand:
            brand_score = fuzz.token_set_ratio(brand, car.get("brand", ""))
            if brand_score >= 80:
                score += W.brand_exact
            elif brand_score >= 60:
                score += W.brand_partial

        # ── Stage 6: Transmission ─────────────────────────────────
        if tx:
            car_txs = [x.lower() for x in car.get("transmission", [])]
            if tx.lower() in car_txs:
                score += W.transmission

        # ── Stage 7: Features ─────────────────────────────────────
        if f.feature_keywords:
            feature_text = " ".join(
                car.get("features", []) + car.get("best_for", [])
            ).lower()
            hits = sum(
                1 for kw in f.feature_keywords
                if fuzz.token_set_ratio(kw.lower(), feature_text) >= 60
            )
            score += min(hits * W.feature_per_hit, W.feature_max)

        # ── Stage 8: Popularity bonus ─────────────────────────────
        popular_count = len(car.get("popular_in", []))
        score += min(popular_count * W.popularity_per, W.popularity_max)

        return score

    @staticmethod
    def _lookup_by_name(
        query: str,
        cars: list[dict],
        max_results: int,
    ) -> list[dict]:
        """
        Short-circuit for specific car name queries.
        Uses fuzzy name match → perfect score of 100.
        """
        names = [c["name"] for c in cars]
        result = rfuzz_process.extractOne(
            query, names,
            scorer=fuzz.token_set_ratio,
            score_cutoff=55,
        )
        if result:
            matched_name = result[0]
            print(f"[FUZZY CARS] Car name '{query}' -> '{matched_name}'")
            return [
                {**c, "_relevance_score": 100.0}
                for c in cars if c["name"] == matched_name
            ][:max_results]
        return []

    @staticmethod
    def _apply_sort(
        scored: list[tuple[dict, float]],
        sort_by: str,
    ) -> list[tuple[dict, float]]:
        """Sort candidate list by the strategy specified in the QueryPlan."""
        if sort_by == "price_asc":
            return sorted(scored, key=lambda x: (x[0]["price_lakh"]["min"], -x[1]))
        if sort_by == "price_desc":
            return sorted(scored, key=lambda x: (-x[0]["price_lakh"]["max"], -x[1]))
        if sort_by == "mileage_desc":
            def _max_mileage(car: dict) -> float:
                m = car.get("mileage_kmpl", {})
                return max(m.values()) if isinstance(m, dict) and m else 0.0
            return sorted(scored, key=lambda x: (-_max_mileage(x[0]), -x[1]))
        if sort_by == "seating_desc":
            return sorted(scored, key=lambda x: (-x[0].get("seating", 5), -x[1]))
        # Default: relevance score
        return sorted(scored, key=lambda x: -x[1])

    @staticmethod
    def _default_service_items(intervals: dict) -> dict:
        """
        Returns the 5 most universal service items as a safe fallback.
        Used when keyword search finds no strong match.
        """
        common = ["engine_oil", "air_filter", "spark_plugs", "brake_pads", "timing_belt"]
        return {k: intervals[k] for k in common if k in intervals}


# ─────────────────────────────────────────────
# Module-level singleton
# Import this everywhere — do not instantiate FuzzySearchEngine directly.
# ─────────────────────────────────────────────
fuzzy_engine = FuzzySearchEngine()
