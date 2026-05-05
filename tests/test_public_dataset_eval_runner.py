# coding=utf-8
"""Tests for model-independent public dataset eval scorers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "public_dataset_eval_runner.py"
SPEC = importlib.util.spec_from_file_location("public_dataset_eval_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_humaneval_scorer_runs_normalized_case() -> None:
    case = {
        "case_id": "HumanEval/0",
        "entry_point": "add",
        "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
    }
    pred = {"case_id": "HumanEval/0", "completion": "def add(a, b):\n    return a + b"}
    out = runner.score_humaneval_case(case, pred, timeout_s=3)
    assert out["ok"] is True


def test_bfcl_scorer_checks_name_and_arguments() -> None:
    case = {
        "case_id": "bfcl_0",
        "expected_tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
        ],
    }
    pred = {
        "case_id": "bfcl_0",
        "tool_calls": [
            {"name": "get_weather", "arguments": {"unit": "celsius", "city": "Paris"}}
        ],
    }
    out = runner.score_bfcl_case(case, pred)
    assert out["ok"] is True
    assert out["schema_valid"] is True
    assert out["function_name_match"] is True
    assert out["arguments_match"] is True


def test_bfcl_scorer_accepts_public_answer_options() -> None:
    case = {
        "case_id": "bfcl_simple_0",
        "expected_tool_call_options": [
            [
                {
                    "name": "calculate_triangle_area",
                    "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
                }
            ]
        ],
    }
    pred = {
        "case_id": "bfcl_simple_0",
        "tool_calls": [
            {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}
        ],
    }
    out = runner.score_bfcl_case(case, pred)
    assert out["ok"] is True
    assert out["arguments_match"] is True


def test_simpleqa_scorer_uses_aliases_and_not_attempted() -> None:
    case = {
        "case_id": "simpleqa_0",
        "answer": "Paris",
        "aliases": ["Paris, France"],
    }
    assert runner.score_simpleqa_case(case, {"answer": "Paris, France"})["ok"] is True
    unsure = runner.score_simpleqa_case(case, {"answer": "I don't know"})
    assert unsure["reason"] == "not_attempted"
    assert unsure["not_attempted"] is True


def test_retrieval_metrics_basic() -> None:
    out = runner.retrieval_metrics({"d2", "d4"}, ["d1", "d2", "d3", "d4"])
    assert out["recall_at_1"] == 0.0
    assert out["recall_at_5"] == 1.0
    assert out["mrr_at_10"] == 0.5
    assert out["ndcg_at_10"] > 0.0
