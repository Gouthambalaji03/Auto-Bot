"""
AutoBot Evals Framework — Production Evaluation Suite
Comprehensive benchmarks for automotive safety, tool precision, intent routing, UI card compliance, and latency.
"""

from evals.metrics import (
    EvaluationResult,
    evaluate_safety,
    evaluate_tools,
    evaluate_intent,
    evaluate_card_format,
    evaluate_response,
)

__all__ = [
    "EvaluationResult",
    "evaluate_safety",
    "evaluate_tools",
    "evaluate_intent",
    "evaluate_card_format",
    "evaluate_response",
]
