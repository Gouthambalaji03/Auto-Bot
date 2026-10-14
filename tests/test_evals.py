"""
tests/test_evals.py — Automated Unit & Integration Tests for the Evals Suite
Works out-of-the-box with both 'pytest' and standard library 'python -m unittest'.
"""

import json
import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evals.metrics import (
    evaluate_safety,
    evaluate_tools,
    evaluate_intent,
    evaluate_card_format,
    evaluate_content,
    evaluate_response,
)


class TestEvalsSuite(unittest.TestCase):

    def test_eval_dataset_integrity(self):
        """Verify that dataset.json exists, is valid JSON, and has unique test IDs."""
        dataset_file = ROOT_DIR / "evals" / "dataset.json"
        self.assertTrue(dataset_file.exists(), "evals/dataset.json must exist")

        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 10, "Dataset should have at least 10 benchmarks")

        ids = [item["id"] for item in data]
        self.assertEqual(len(ids), len(set(ids)), "Test IDs in dataset.json must be unique")

        for item in data:
            self.assertIn("id", item)
            self.assertIn("category", item)
            self.assertIn("query", item)
            self.assertIn("name", item)

    def test_safety_metric_passes_on_triage_warning(self):
        """Safety metric should pass when 'do not drive' is properly stated."""
        test_case = {
            "id": "test_safe",
            "category": "safety",
            "must_contain": ["do not drive"],
            "must_not_contain": ["safe to drive"],
        }
        content = "⚠️ Critical Warning: Do NOT drive this vehicle. Arrange a flatbed tow immediately."
        score, failures = evaluate_safety(test_case, content)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(failures), 0)

    def test_safety_metric_fails_on_dangerous_advice(self):
        """Safety metric should flag missing warning or forbidden permissive terms."""
        test_case = {
            "id": "test_safe_fail",
            "category": "safety",
            "must_contain": ["do not drive"],
            "must_not_contain": ["safe to drive"],
        }
        content = "It is safe to drive home slowly if you pump the brakes gently."
        score, failures = evaluate_safety(test_case, content)
        self.assertEqual(score, 0.0)
        self.assertGreaterEqual(len(failures), 1)

    def test_tool_calling_metric(self):
        """Verify tool precision evaluator."""
        test_case = {
            "expected_tools": ["calculate_loan_emi"],
        }
        score, failures = evaluate_tools(test_case, ["calculate_loan_emi"])
        self.assertEqual(score, 1.0)
        self.assertEqual(len(failures), 0)

        score_fail, failures_fail = evaluate_tools(test_case, ["search_catalog"])
        self.assertEqual(score_fail, 0.0)
        self.assertEqual(len(failures_fail), 1)

    def test_card_format_metric(self):
        """Verify car card DOM structure detection."""
        test_case = {"must_produce_cards": True}
        valid_card_markdown = """
### 1. Tata Nexon EV · Compact SUV EV (₹14.49 L – ₹19.49 L)
* 💰 **Price:** ₹14.49 L – ₹19.49 L
* ⛽ **Fuel & Engine:** Electric (EV)
* 📊 **Range:** 465 km
* ⚙️ **Transmission:** Automatic
* 🌟 **Key Features:** ADAS, Fast Charging

> 💡 **Best For:** City commuting and low running cost.
"""
        score, failures = evaluate_card_format(test_case, valid_card_markdown)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(failures), 0)


if __name__ == "__main__":
    unittest.main()
