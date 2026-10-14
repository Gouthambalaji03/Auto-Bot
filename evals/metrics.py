"""
evals/metrics.py — Production Evaluation Metrics for AutoBot

Implements deterministic evaluation metrics for automotive safety,
tool calling precision, intent classification, Car Card DOM rendering,
and keyword compliance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from ui.app import format_car_recommendation_cards
except Exception:
    def format_car_recommendation_cards(text: str) -> str:
        if not text or '<div class="car-card"' in text or "<div class='car-card'" in text:
            return text
        pattern = re.compile(
            r"###\s+(?:(?:\d+\.|\*|\-)\s+)?([^\n·\(\)]+?)(?:\s+[·\-–]\s+([^\n\(\)]+?))?(?:\s*\(([^\n\)]+?)\))?\n\n?"
            r"((?:[\*\-]\s+[^\n]+\n?)+)"
            r"(?:\n*>+\s*(?:💡\s*)?\*\*Best For:\*\*\s*([^\n]+))?",
            re.MULTILINE
        )
        def replace_with_card(match):
            name = match.group(1).strip()
            segment = (match.group(2) or "").strip()
            price = (match.group(3) or "").strip()
            return f'<div class="car-card"><div class="car-card-header"><h3 class="car-name">{name}</h3><span class="car-segment-badge">{segment}</span><div class="car-price-badge">{price}</div></div><div class="car-specs-grid"></div></div>'
        return pattern.sub(replace_with_card, text)


@dataclass
class EvaluationResult:
    test_id: str
    category: str
    name: str
    passed: bool
    score: float  # Composite 0.0 to 1.0
    metric_scores: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    cached_tokens: int = 0
    actual_tools: list[str] = field(default_factory=list)
    actual_intents: list[str] = field(default_factory=list)
    response_snippet: str = ""


def evaluate_safety(test_case: dict[str, Any], content: str) -> tuple[float, list[str]]:
    """Evaluates automotive safety compliance and do-not-drive triage rules."""
    failures = []
    text_lower = content.lower()

    # If test specifies must_contain for safety
    for phrase in test_case.get("must_contain", []):
        if phrase.lower() not in text_lower:
            failures.append(f"Safety failure: missing mandatory phrase '{phrase}'")

    # Check forbidden dangerous phrases
    for forbidden in test_case.get("must_not_contain", []):
        if forbidden.lower() in text_lower:
            failures.append(f"Safety failure: contains forbidden phrase '{forbidden}'")

    score = 1.0 if not failures else 0.0
    return score, failures


def evaluate_tools(test_case: dict[str, Any], actual_tools: list[str]) -> tuple[float, list[str]]:
    """Evaluates whether the agent invoked the expected tools."""
    expected_tools = test_case.get("expected_tools", [])
    if not expected_tools:
        return 1.0, []

    failures = []
    # If any tool from expected list was called, tool selection succeeded
    matches = [t for t in expected_tools if t in actual_tools]
    if not matches:
        failures.append(f"Tool failure: expected at least one of {expected_tools}, got {actual_tools}")
        return 0.0, failures

    score = min(1.0, len(matches) / len(expected_tools))
    return score, failures


def evaluate_intent(test_case: dict[str, Any], actual_intents: list[str]) -> tuple[float, list[str]]:
    """Evaluates LLM intent classification correctness."""
    expected = test_case.get("expected_intents", [])
    if not expected:
        return 1.0, []

    failures = []
    matched = [exp for exp in expected if exp in actual_intents]
    if not matched:
        failures.append(f"Intent failure: expected {expected}, got {actual_intents}")
        return 0.0, failures

    score = len(matched) / len(expected)
    return score, failures


def evaluate_card_format(test_case: dict[str, Any], content: str) -> tuple[float, list[str]]:
    """Evaluates whether vehicle recommendations convert properly to HTML car cards."""
    if not test_case.get("must_produce_cards", False):
        return 1.0, []

    failures = []
    formatted = format_car_recommendation_cards(content)
    card_count = formatted.count('<div class="car-card">')

    if card_count == 0:
        failures.append("Card format failure: output failed to produce any '<div class=\"car-card\">' elements")
        return 0.0, failures

    # Must contain car-specs-grid
    if '<div class="car-specs-grid">' not in formatted:
        failures.append("Card format failure: car-card is missing specs grid")
        return 0.5, failures

    return 1.0, []


def evaluate_content(test_case: dict[str, Any], content: str) -> tuple[float, list[str]]:
    """Evaluates keyword requirements, any_of matches, and forbidden terms."""
    failures = []
    text_lower = content.lower()

    # must_contain (if not already safety checked)
    if test_case.get("category") != "safety":
        for phrase in test_case.get("must_contain", []):
            if phrase.lower() not in text_lower:
                failures.append(f"Content failure: missing required string '{phrase}'")

    # any_of
    any_of = test_case.get("any_of", [])
    if any_of:
        if not any(term.lower() in text_lower for term in any_of):
            failures.append(f"Content failure: must match at least one of {any_of}")

    # must_not_contain
    if test_case.get("category") != "safety":
        for forbidden in test_case.get("must_not_contain", []):
            if forbidden.lower() in text_lower:
                failures.append(f"Content failure: output contains forbidden term '{forbidden}'")

    score = 1.0 if not failures else 0.0
    return score, failures


def evaluate_response(
    test_case: dict[str, Any],
    content: str,
    actual_intents: list[str],
    actual_tools: list[str],
    latency_seconds: float = 0.0,
    cached_tokens: int = 0,
) -> EvaluationResult:
    """Aggregates all metrics and returns a standardized EvaluationResult."""
    all_failures: list[str] = []
    metric_scores: dict[str, float] = {}

    # Safety
    safety_score, safety_failures = evaluate_safety(test_case, content)
    metric_scores["safety"] = safety_score
    all_failures.extend(safety_failures)

    # Tool calls
    tool_score, tool_failures = evaluate_tools(test_case, actual_tools)
    metric_scores["tools"] = tool_score
    all_failures.extend(tool_failures)

    # Intent
    intent_score, intent_failures = evaluate_intent(test_case, actual_intents)
    metric_scores["intent"] = intent_score
    all_failures.extend(intent_failures)

    # Card Format
    card_score, card_failures = evaluate_card_format(test_case, content)
    metric_scores["card_format"] = card_score
    all_failures.extend(card_failures)

    # Content
    content_score, content_failures = evaluate_content(test_case, content)
    metric_scores["content"] = content_score
    all_failures.extend(content_failures)

    # Latency penalty if query took > 15 seconds
    if latency_seconds > 15.0:
        all_failures.append(f"Performance warning: latency was {latency_seconds:.1f}s (> 15s)")

    # Composite score (average of all active metrics)
    active_scores = list(metric_scores.values())
    composite = sum(active_scores) / len(active_scores) if active_scores else 0.0

    passed = len(all_failures) == 0

    return EvaluationResult(
        test_id=test_case["id"],
        category=test_case["category"],
        name=test_case["name"],
        passed=passed,
        score=round(composite, 2),
        metric_scores=metric_scores,
        failures=all_failures,
        latency_seconds=latency_seconds,
        cached_tokens=cached_tokens,
        actual_tools=actual_tools,
        actual_intents=actual_intents,
        response_snippet=content[:200].replace("\n", " ") + ("..." if len(content) > 200 else ""),
    )
