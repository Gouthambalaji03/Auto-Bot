"""
evals/runner.py — Production Evaluation Runner for AutoBot

Executes benchmark suites against the active AutoBot agent, scores
performance against safety, tools, intent, card formatting, and latency,
and exports timestamped evaluation reports.

Usage:
    python -m evals.runner
    python -m evals.runner --category safety
    python -m evals.runner --category finance
    python -m evals.runner --id safety_01
    python -m evals.runner --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env", override=True)

from agents.automotive_agent import stream_chat_with_autobot
from evals.metrics import EvaluationResult, evaluate_response


async def run_single_eval(test_case: dict[str, Any]) -> EvaluationResult:
    """Executes a single test case through AutoBot and returns the scored evaluation."""
    user_query = test_case["query"]
    final_content = ""
    actual_intents: list[str] = []
    actual_tools: list[str] = []
    elapsed_seconds = 0.0
    cached_tokens = 0

    started = time.monotonic()
    try:
        async for update in stream_chat_with_autobot(user_query):
            final_content = update.content
            if update.complete:
                actual_intents = list(update.intents)
                actual_tools = list(update.tool_calls)
                elapsed_seconds = update.elapsed_seconds
                cached_tokens = update.cached_tokens
    except Exception as exc:
        elapsed_seconds = round(time.monotonic() - started, 2)
        return EvaluationResult(
            test_id=test_case["id"],
            category=test_case["category"],
            name=test_case["name"],
            passed=False,
            score=0.0,
            failures=[f"Execution exception: {type(exc).__name__}: {exc}"],
            latency_seconds=elapsed_seconds,
            response_snippet="ERROR EXECUTING QUERY",
        )

    return evaluate_response(
        test_case=test_case,
        content=final_content,
        actual_intents=actual_intents,
        actual_tools=actual_tools,
        latency_seconds=elapsed_seconds,
        cached_tokens=cached_tokens,
    )


async def run_eval_suite(
    category_filter: str | None = None,
    test_id_filter: str | None = None,
    limit: int | None = None,
) -> list[EvaluationResult]:
    """Loads dataset and executes evaluations."""
    dataset_path = ROOT_DIR / "evals" / "dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    # Filtering
    if category_filter:
        dataset = [t for t in dataset if t.get("category") == category_filter]
    if test_id_filter:
        dataset = [t for t in dataset if t.get("id") == test_id_filter]
    if limit:
        dataset = dataset[:limit]

    total = len(dataset)
    print("=" * 70)
    print("  🚗 AUTOBOT PRODUCTION EVALUATION SUITE")
    print(f"  Target: {total} benchmarks | Model: {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results: list[EvaluationResult] = []

    for index, test in enumerate(dataset, start=1):
        test_id = test["id"]
        category = test.get("category", "general")
        name = test.get("name", test_id)

        print(f"\n[{index}/{total}] Running: [{category.upper()}] {name} ({test_id})...", flush=True)

        res = await run_single_eval(test)
        results.append(res)

        status_icon = "✅ PASS" if res.passed else "❌ FAIL"
        tools_repr = f"Tools: {res.actual_tools}" if res.actual_tools else "Tools: (none)"
        cache_repr = f"Cache: {res.cached_tokens} tok" if res.cached_tokens > 0 else "Cache: miss"

        print(f"       {status_icon} | Score: {res.score:.2f} | Latency: {res.latency_seconds:.2f}s | {cache_repr}")
        print(f"       {tools_repr}")

        if not res.passed:
            for failure in res.failures:
                print(f"       ⚠️  {failure}")

    # Generate summary report
    print_summary(results)
    save_reports(results)
    return results


def print_summary(results: list[EvaluationResult]) -> None:
    """Prints a formatted evaluation scorecard in the terminal."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0.0

    avg_latency = sum(r.latency_seconds for r in results) / total if total > 0 else 0.0
    total_cached = sum(r.cached_tokens for r in results)

    # Category breakdown
    categories: dict[str, list[EvaluationResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print("\n" + "=" * 70)
    print("  📊 EVALUATION BENCHMARK SCORECARD")
    print("=" * 70)
    print(f"  Overall Result:     {'✅ PASSED' if failed == 0 else '⚠️ ISSUES DETECTED'}")
    print(f"  Pass Rate:          {passed}/{total} ({pass_rate:.1f}%)")
    print(f"  Average Latency:    {avg_latency:.2f}s per query")
    print(f"  Total Cached Tokens:{total_cached:,} tokens saved")
    print("-" * 70)
    print(f"  {'CATEGORY':<20} | {'TOTAL':<6} | {'PASSED':<7} | {'ACCURACY':<8}")
    print("-" * 70)

    for cat, cat_res in sorted(categories.items()):
        c_tot = len(cat_res)
        c_pass = sum(1 for r in cat_res if r.passed)
        c_rate = (c_pass / c_tot * 100) if c_tot > 0 else 0.0
        print(f"  {cat:<20} | {c_tot:<6} | {c_pass:<7} | {c_rate:>6.1f}%")

    print("=" * 70)


def save_reports(results: list[EvaluationResult]) -> None:
    """Exports structured JSON and Markdown reports to evals/reports/."""
    reports_dir = ROOT_DIR / "evals" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"eval_report_{timestamp}.json"
    md_path = reports_dir / f"eval_report_{timestamp}.md"

    # Save JSON report
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "pass_rate_percentage": round(sum(1 for r in results if r.passed) / max(len(results), 1) * 100, 2),
        "results": [
            {
                "test_id": r.test_id,
                "category": r.category,
                "name": r.name,
                "passed": r.passed,
                "score": r.score,
                "latency_seconds": r.latency_seconds,
                "cached_tokens": r.cached_tokens,
                "actual_tools": r.actual_tools,
                "actual_intents": r.actual_intents,
                "failures": r.failures,
                "metric_scores": r.metric_scores,
                "response_snippet": r.response_snippet,
            }
            for r in results
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown report
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# AutoBot Evaluation Report — {timestamp}\n\n")
        f.write(f"- **Pass Rate:** {report_data['passed']}/{report_data['total']} ({report_data['pass_rate_percentage']}%)\n")
        f.write(f"- **Model:** `{report_data['model']}`\n")
        f.write(f"- **Date:** {report_data['timestamp']}\n\n")
        f.write("## Test Breakdown\n\n")
        f.write("| ID | Category | Name | Status | Score | Latency |\n")
        f.write("|:---|:---|:---|:---:|:---:|:---:|\n")
        for r in results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            f.write(f"| `{r.test_id}` | {r.category} | {r.name} | {status} | {r.score} | {r.latency_seconds:.2f}s |\n")

    print(f"\n[REPORT] Saved evaluation report to:\n  - {json_path}\n  - {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoBot Production Evaluation Suite")
    parser.add_argument("--category", type=str, help="Filter by category (safety, finance, buying, etc.)")
    parser.add_argument("--id", type=str, help="Filter by single test case ID")
    parser.add_argument("--limit", type=int, help="Limit number of test queries")
    args = parser.parse_args()

    asyncio.run(
        run_eval_suite(
            category_filter=args.category,
            test_id_filter=args.id,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
