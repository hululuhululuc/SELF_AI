# coding=utf-8
"""Model-independent public dataset evaluation protocol for Self AI.

This runner deliberately separates scoring from model generation:

- HumanEval+ mini: consumes normalized coding cases + prediction JSONL and
  executes provided tests in a subprocess.
- BFCL mini: consumes normalized function-calling cases + prediction JSONL and
  checks tool-call schema/name/argument correctness without executing tools.
- SimpleQA mini: consumes normalized factual QA cases + prediction JSONL and
  applies deterministic exact/alias grading.
- SciFact retrieval mini: indexes BEIR-style SciFact files into Qdrant with
  local BGE-M3 and measures retrieval metrics directly.

No Self AI model calls happen in this file. Generation adapters can be added
later and should write the prediction JSONL consumed here.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RUNS_ROOT = ROOT_DIR / "benchmarks" / "runs"

STAGE_DEFAULT_LIMITS: dict[str, int] = {
    "humaneval_plus": 5,
    "bfcl": 10,
    "simpleqa": 10,
    "scifact_retrieval": 10,
}

STOP_REASONS = {
    "unhandled_exception",
    "runner_error",
    "missing_prediction",
    "invalid_case",
    "scifact_index_failed",
    "scifact_search_failed",
}

NOT_ATTEMPTED_PATTERNS = (
    "i don't know",
    "i dont know",
    "i do not know",
    "not sure",
    "cannot answer",
    "can't answer",
    "cant answer",
    "unknown",
)


@dataclass(frozen=True)
class EvalPaths:
    run_dir: Path
    cases_dir: Path
    config_path: Path
    preflight_path: Path
    subset_manifest_path: Path
    raw_predictions_path: Path
    metrics_path: Path
    failures_path: Path
    latency_trace_path: Path
    report_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix)
    return f"{stamp}_{safe or 'public_dataset_eval'}"


def _make_paths(run_name: str) -> EvalPaths:
    run_dir = RUNS_ROOT / run_name
    return EvalPaths(
        run_dir=run_dir,
        cases_dir=run_dir / "cases",
        config_path=run_dir / "config.json",
        preflight_path=run_dir / "preflight.json",
        subset_manifest_path=run_dir / "subset_manifest.json",
        raw_predictions_path=run_dir / "raw_predictions.jsonl",
        metrics_path=run_dir / "metrics.json",
        failures_path=run_dir / "failures.json",
        latency_trace_path=run_dir / "latency_trace.jsonl",
        report_path=run_dir / "report.md",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL row must be object at {path}:{line_no}")
            rows.append(parsed)
    return rows


def _load_cases(path: Path, *, stage: str, limit: int, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"cases file not found: {path}")
    rows = _read_jsonl(path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        case_stage = str(row.get("stage", stage) or stage)
        if case_stage == stage:
            selected.append(row)
    start = max(0, int(offset or 0))
    selected = selected[start:]
    if limit:
        selected = selected[: max(1, int(limit))]
    return selected


def _effective_limit(args: argparse.Namespace, stage: str) -> int:
    values = [
        int(x)
        for x in (getattr(args, "batch_size", 0), getattr(args, "limit", 0))
        if int(x or 0) > 0
    ]
    if values:
        return max(1, min(values))
    return STAGE_DEFAULT_LIMITS[stage]


def _load_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"predictions file not found: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        case_id = str(row.get("case_id") or row.get("task_id") or row.get("id") or "")
        if not case_id:
            raise ValueError("prediction row missing case_id/task_id/id")
        out[case_id] = row
    return out


def _case_id(case: dict[str, Any]) -> str:
    return str(case.get("case_id") or case.get("task_id") or case.get("id") or "")


def _safe_case_dir_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("_") or "case"


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[^\w\s.-]", "", text)
    return text.strip()


def _canonical_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical_json(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json(x) for x in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def _values_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if actual is None and expected == "":
        return True
    if actual == "" and expected is None:
        return True
    return _normalize_text(actual) == _normalize_text(expected)


def _extract_python_code(prediction: dict[str, Any]) -> str:
    for key in ("code", "completion", "response", "answer"):
        text = str(prediction.get(key, "") or "").strip()
        if text:
            break
    else:
        return ""
    fenced = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _validate_humaneval_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _case_id(case):
        errors.append("case_id_missing")
    if not str(case.get("entry_point", "") or "").strip():
        errors.append("entry_point_missing")
    if not str(case.get("test_code", "") or "").strip():
        errors.append("test_code_missing")
    return errors


def _run_python_case(
    *,
    code: str,
    test_code: str,
    entry_point: str,
    timeout_s: float,
) -> dict[str, Any]:
    if not code.strip():
        return {"passed": False, "reason": "empty_code", "stdout": "", "stderr": ""}
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {
            "passed": False,
            "reason": "syntax_error",
            "stdout": "",
            "stderr": f"{exc.__class__.__name__}: {exc}",
        }

    runner = (
        code
        + "\n\n"
        + test_code
        + "\n\n"
        + "if __name__ == '__main__':\n"
        + f"    check({entry_point})\n"
    )
    with tempfile.TemporaryDirectory(prefix="self_ai_eval_") as td:
        path = Path(td) / "candidate_eval.py"
        path.write_text(runner, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=td,
                text=True,
                capture_output=True,
                timeout=max(1.0, float(timeout_s)),
            )
            return {
                "passed": proc.returncode == 0,
                "reason": "" if proc.returncode == 0 else "test_failed",
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "reason": "timeout",
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or "")[-4000:],
            }


def score_humaneval_case(
    case: dict[str, Any],
    prediction: dict[str, Any] | None,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    errors = _validate_humaneval_case(case)
    if errors:
        return {"ok": False, "reason": "invalid_case", "details": errors}
    if prediction is None:
        return {"ok": False, "reason": "missing_prediction", "details": ""}
    code = _extract_python_code(prediction)
    test_result = _run_python_case(
        code=code,
        test_code=str(case.get("test_code") or ""),
        entry_point=str(case.get("entry_point") or ""),
        timeout_s=timeout_s,
    )
    return {
        "ok": bool(test_result.get("passed", False)),
        "reason": "" if test_result.get("passed") else str(test_result.get("reason", "test_failed")),
        "details": test_result,
        "syntax_ok": str(test_result.get("reason")) != "syntax_error",
    }


def _coerce_tool_calls(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        return _coerce_tool_calls(parsed)
    if isinstance(value, dict):
        if "tool_calls" in value:
            return _coerce_tool_calls(value.get("tool_calls"))
        if "calls" in value:
            return _coerce_tool_calls(value.get("calls"))
        name = value.get("name") or value.get("function") or value.get("tool_name")
        args = value.get("arguments", value.get("args", {}))
        if isinstance(name, str) and name.strip():
            return [{"name": name.strip(), "arguments": args if isinstance(args, dict) else {}}]
        return []
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            out.extend(_coerce_tool_calls(item))
        return out
    return []


def _prediction_tool_calls(prediction: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    raw = (
        prediction.get("tool_calls")
        if "tool_calls" in prediction
        else prediction.get("calls", prediction.get("response"))
    )
    calls = _coerce_tool_calls(raw)
    return calls, bool(calls)


def _expected_tool_calls(case: dict[str, Any]) -> list[dict[str, Any]]:
    raw = case.get("expected_tool_calls", case.get("expected_calls", []))
    return _coerce_tool_calls(raw)


def _expected_tool_call_options(case: dict[str, Any]) -> list[list[dict[str, Any]]]:
    raw_options = case.get("expected_tool_call_options")
    if isinstance(raw_options, list) and raw_options:
        options: list[list[dict[str, Any]]] = []
        for option in raw_options:
            calls = _coerce_tool_calls(option)
            if calls:
                options.append(calls)
        if options:
            return options
    expected = _expected_tool_calls(case)
    return [expected] if expected else []


def _argument_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_argument_value_matches(actual, item) for item in expected)
    return _values_equal(actual, expected)


def _arguments_match_expected(predicted_args: dict[str, Any], expected_args: dict[str, Any]) -> bool:
    if not isinstance(predicted_args, dict) or not isinstance(expected_args, dict):
        return False
    for key, expected_value in expected_args.items():
        if key not in predicted_args:
            if _argument_value_matches("", expected_value):
                continue
            return False
        if not _argument_value_matches(predicted_args.get(key), expected_value):
            return False
    for key, predicted_value in predicted_args.items():
        if key not in expected_args and str(predicted_value or "").strip():
            return False
    return True


def _tool_calls_match_expected(predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> tuple[bool, bool]:
    expected_names = [str(x.get("name", "")) for x in expected]
    predicted_names = [str(x.get("name", "")) for x in predicted]
    name_match = predicted_names == expected_names
    if not name_match or len(predicted) != len(expected):
        return name_match, False
    args_match = all(
        _arguments_match_expected(
            predicted_call.get("arguments", {}) if isinstance(predicted_call.get("arguments", {}), dict) else {},
            expected_call.get("arguments", {}) if isinstance(expected_call.get("arguments", {}), dict) else {},
        )
        for predicted_call, expected_call in zip(predicted, expected, strict=False)
    )
    return name_match, args_match


def score_bfcl_case(case: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    if not _case_id(case):
        return {"ok": False, "reason": "invalid_case", "details": ["case_id_missing"]}
    expected_options = _expected_tool_call_options(case)
    if not expected_options:
        return {"ok": False, "reason": "invalid_case", "details": ["expected_tool_calls_missing"]}
    if prediction is None:
        return {"ok": False, "reason": "missing_prediction", "details": ""}
    predicted, schema_valid = _prediction_tool_calls(prediction)
    if not schema_valid:
        return {
            "ok": False,
            "reason": "invalid_tool_call_schema",
            "details": {"expected_options": expected_options, "predicted": predicted},
            "schema_valid": False,
            "function_name_match": False,
            "arguments_match": False,
        }
    expected = expected_options[0]
    expected_names = [str(x.get("name", "")) for x in expected]
    predicted_names = [str(x.get("name", "")) for x in predicted]
    option_results = []
    name_match = False
    args_match = False
    matched_expected: list[dict[str, Any]] | None = None
    for option in expected_options:
        option_name_match, option_args_match = _tool_calls_match_expected(predicted, option)
        option_results.append(
            {
                "expected_names": [str(x.get("name", "")) for x in option],
                "function_name_match": option_name_match,
                "arguments_match": option_args_match,
            }
        )
        name_match = name_match or option_name_match
        if option_name_match and option_args_match:
            args_match = True
            matched_expected = option
            break
    return {
        "ok": bool(name_match and args_match),
        "reason": "" if name_match and args_match else "tool_call_mismatch",
        "details": {
            "expected": matched_expected or expected,
            "expected_options": expected_options,
            "predicted": predicted,
            "expected_names": expected_names,
            "predicted_names": predicted_names,
            "option_results": option_results,
        },
        "schema_valid": True,
        "function_name_match": name_match,
        "arguments_match": args_match,
    }


def score_simpleqa_case(case: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    if not _case_id(case):
        return {"ok": False, "reason": "invalid_case", "details": ["case_id_missing"]}
    answers = [case.get("answer", "")]
    aliases = case.get("aliases", [])
    if isinstance(aliases, list):
        answers.extend(aliases)
    normalized_answers = {_normalize_text(x) for x in answers if _normalize_text(x)}
    if not normalized_answers:
        return {"ok": False, "reason": "invalid_case", "details": ["answer_missing"]}
    if prediction is None:
        return {"ok": False, "reason": "missing_prediction", "details": ""}
    raw_answer = str(
        prediction.get("answer")
        or prediction.get("response")
        or prediction.get("prediction")
        or ""
    ).strip()
    normalized_prediction = _normalize_text(raw_answer)
    not_attempted = any(pat in normalized_prediction for pat in NOT_ATTEMPTED_PATTERNS)
    if not_attempted:
        return {
            "ok": False,
            "reason": "not_attempted",
            "details": {"prediction": raw_answer, "accepted": sorted(normalized_answers)},
            "not_attempted": True,
            "incorrect": False,
        }
    correct = normalized_prediction in normalized_answers
    return {
        "ok": correct,
        "reason": "" if correct else "incorrect",
        "details": {"prediction": raw_answer, "accepted": sorted(normalized_answers)},
        "not_attempted": False,
        "incorrect": not correct,
    }


def _dcg(relevance: list[int]) -> float:
    total = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            total += 1.0 / math.log2(idx + 1)
    return total


def retrieval_metrics(relevant_ids: set[str], ranked_ids: list[str], *, k_values: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not relevant_ids:
        for k in k_values:
            out[f"recall_at_{k}"] = 0.0
        out["mrr_at_10"] = 0.0
        out["ndcg_at_10"] = 0.0
        return out
    for k in k_values:
        top = ranked_ids[:k]
        out[f"recall_at_{k}"] = round(len(set(top) & relevant_ids) / len(relevant_ids), 6)
    reciprocal = 0.0
    for idx, doc_id in enumerate(ranked_ids[:10], start=1):
        if doc_id in relevant_ids:
            reciprocal = 1.0 / idx
            break
    out["mrr_at_10"] = round(reciprocal, 6)
    gains = [1 if doc_id in relevant_ids else 0 for doc_id in ranked_ids[:10]]
    ideal = [1] * min(len(relevant_ids), 10)
    denom = _dcg(ideal)
    out["ndcg_at_10"] = round(_dcg(gains) / denom, 6) if denom else 0.0
    return out


def _average_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        validation = row.get("validation", {}) if isinstance(row.get("validation"), dict) else {}
        try:
            values.append(float(validation.get(key, 0.0) or 0.0))
        except Exception:
            pass
    return round(sum(values) / len(values), 6) if values else 0.0


def _failure_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("ok"):
            continue
        validation = row.get("validation", {}) if isinstance(row.get("validation"), dict) else {}
        reason = str(validation.get("reason", "unknown") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _latency_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(float(row.get("elapsed_s", 0) or 0) for row in rows)
    if not values:
        return {"latency_avg_s": 0.0, "latency_p50_s": 0.0, "latency_p95_s": 0.0}
    def pct(p: float) -> float:
        if len(values) == 1:
            return values[0]
        idx = (len(values) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(values) - 1)
        w = idx - lo
        return values[lo] * (1 - w) + values[hi] * w
    return {
        "latency_avg_s": round(sum(values) / len(values), 3),
        "latency_p50_s": round(pct(0.5), 3),
        "latency_p95_s": round(pct(0.95), 3),
    }


def summarize_rows(stage: str, rows: list[dict[str, Any]], *, selected_case_count: int, offset: int, batch_size: int, stopped_early: bool) -> dict[str, Any]:
    total = len(rows)
    ok_count = sum(1 for row in rows if bool(row.get("ok", False)))
    metrics: dict[str, Any] = {
        "schema_version": "public_dataset_eval.v1",
        "stage": stage,
        "generated_at": _utc_now(),
        "total_cases": total,
        "ok_cases": ok_count,
        "failed_cases": total - ok_count,
        "ok_rate": round(ok_count / total, 6) if total else 0.0,
        "selected_case_count": selected_case_count,
        "executed_case_count": total,
        "offset": offset,
        "batch_size": batch_size,
        "stopped_early": stopped_early,
        "failure_reasons": _failure_reasons(rows),
        **_latency_stats(rows),
    }
    if stage == "humaneval_plus":
        syntax_ok = sum(1 for row in rows if (row.get("validation") or {}).get("syntax_ok", False))
        metrics["pass_at_1"] = metrics["ok_rate"]
        metrics["syntax_pass_rate"] = round(syntax_ok / total, 6) if total else 0.0
        metrics["timeout_rate"] = round(
            sum(1 for row in rows if (row.get("validation") or {}).get("reason") == "timeout") / total,
            6,
        ) if total else 0.0
    elif stage == "bfcl":
        metrics["schema_valid_rate"] = round(
            sum(1 for row in rows if (row.get("validation") or {}).get("schema_valid", False)) / total,
            6,
        ) if total else 0.0
        metrics["function_name_accuracy"] = round(
            sum(1 for row in rows if (row.get("validation") or {}).get("function_name_match", False)) / total,
            6,
        ) if total else 0.0
        metrics["argument_exact_match"] = round(
            sum(1 for row in rows if (row.get("validation") or {}).get("arguments_match", False)) / total,
            6,
        ) if total else 0.0
    elif stage == "simpleqa":
        metrics["correct_rate"] = metrics["ok_rate"]
        metrics["not_attempted_rate"] = round(
            sum(1 for row in rows if (row.get("validation") or {}).get("not_attempted", False)) / total,
            6,
        ) if total else 0.0
        metrics["incorrect_rate"] = round(
            sum(1 for row in rows if (row.get("validation") or {}).get("incorrect", False)) / total,
            6,
        ) if total else 0.0
    elif stage == "scifact_retrieval":
        for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"):
            metrics[key] = _average_metric(rows, key)
    return metrics


def _build_report(run_name: str, metrics: dict[str, Any], failures: list[dict[str, Any]], paths: EvalPaths) -> str:
    lines = [
        "# Self AI Public Dataset Eval Report",
        "",
        f"Run: `{run_name}`",
        f"Stage: `{metrics.get('stage', '')}`",
        f"Generated: `{metrics.get('generated_at', '')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        if key in {"schema_version", "stage", "generated_at", "failure_reasons"}:
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Config: `{paths.config_path.name}`",
            f"- Preflight: `{paths.preflight_path.name}`",
            f"- Subset manifest: `{paths.subset_manifest_path.name}`",
            f"- Raw predictions: `{paths.raw_predictions_path.name}`",
            f"- Metrics: `{paths.metrics_path.name}`",
            f"- Failures: `{paths.failures_path.name}`",
            f"- Latency trace: `{paths.latency_trace_path.name}`",
            "- Per-case details: `cases/`",
            "",
            "## Failures",
            "",
        ]
    )
    if not failures:
        lines.append("No failed cases.")
    else:
        lines.extend(["| Case | Reason | Details |", "|---|---|---|"])
        for row in failures:
            validation = row.get("validation", {}) if isinstance(row.get("validation"), dict) else {}
            details = str(validation.get("details", ""))[:180].replace("\n", " ")
            lines.append(f"| `{row.get('case_id', '')}` | `{validation.get('reason', '')}` | {details} |")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a lightweight fixed-subset evaluation protocol. Do not present micro-subset scores as official leaderboard submissions.",
            "",
        ]
    )
    return "\n".join(lines)


async def _score_prediction_stage(args: argparse.Namespace, paths: EvalPaths) -> list[dict[str, Any]]:
    stage = str(args.stage)
    limit = _effective_limit(args, stage)
    cases = _load_cases(Path(args.cases_path), stage=stage, limit=limit, offset=int(args.offset or 0))
    predictions = _load_predictions(Path(args.predictions_path) if args.predictions_path else None)
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        case_id = _case_id(case)
        prediction = predictions.get(case_id)
        try:
            if stage == "humaneval_plus":
                validation = score_humaneval_case(case, prediction, timeout_s=float(args.timeout_s))
            elif stage == "bfcl":
                validation = score_bfcl_case(case, prediction)
            elif stage == "simpleqa":
                validation = score_simpleqa_case(case, prediction)
            else:
                raise ValueError(f"unsupported prediction stage: {stage}")
        except Exception as exc:
            validation = {
                "ok": False,
                "reason": "runner_error",
                "details": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            }
        elapsed_s = time.perf_counter() - started
        row = {
            "case_id": case_id,
            "stage": stage,
            "elapsed_s": round(elapsed_s, 3),
            "ok": bool(validation.get("ok", False)),
            "validation": validation,
        }
        case_dir = paths.cases_dir / _safe_case_dir_name(case_id)
        _write_json(case_dir / "input.json", case)
        _write_json(case_dir / "prediction.json", prediction or {})
        _write_json(case_dir / "validation.json", validation)
        _write_json(case_dir / "summary.json", row)
        _append_jsonl(paths.raw_predictions_path, {"case_id": case_id, "input": case, "prediction": prediction or {}, "validation": validation})
        _append_jsonl(paths.latency_trace_path, {"case_id": case_id, "elapsed_s": round(elapsed_s, 3)})
        rows.append(row)
        if bool(args.stop_on_failure) and not row["ok"] and str(validation.get("reason", "")) in STOP_REASONS:
            break
    return rows


def _load_beir_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        doc_id = str(row.get("_id") or row.get("id") or "")
        if doc_id:
            out[doc_id] = row
    return out


def _load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            parts = re.split(r"\s+", text)
            if line_no == 1 and any(x.lower() in {"query-id", "query_id", "corpus-id", "corpus_id"} for x in parts):
                continue
            if len(parts) < 3:
                continue
            qid, doc_id = parts[0], parts[1]
            try:
                score = int(float(parts[2]))
            except Exception:
                score = 1
            if score > 0:
                qrels.setdefault(str(qid), set()).add(str(doc_id))
    return qrels


def _discover_scifact_files(root: Path) -> tuple[Path, Path, Path]:
    corpus = root / "corpus.jsonl"
    queries = root / "queries.jsonl"
    qrels = root / "qrels" / "test.tsv"
    if not corpus.exists():
        corpus = root / "corpus" / "corpus.jsonl"
    if not queries.exists():
        queries = root / "queries" / "queries.jsonl"
    if not qrels.exists():
        candidates = list(root.glob("**/*qrels*.tsv")) + list(root.glob("**/test.tsv"))
        qrels = candidates[0] if candidates else qrels
    return corpus, queries, qrels


async def _score_scifact(args: argparse.Namespace, paths: EvalPaths) -> list[dict[str, Any]]:
    from qdrant_client import QdrantClient

    from self_ai.config import settings
    from self_ai.memory.qdrant_store import QdrantMemory
    from self_ai.schemas import EvidenceItem
    from self_ai.tools import _encode_text

    root = Path(args.scifact_dir)
    corpus_path, queries_path, qrels_path = _discover_scifact_files(root)
    if not corpus_path.exists() or not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(
            f"SciFact files missing. Expected corpus={corpus_path}, queries={queries_path}, qrels={qrels_path}"
        )
    corpus = _load_beir_jsonl(corpus_path)
    queries = _load_beir_jsonl(queries_path)
    qrels = _load_qrels(qrels_path)
    query_ids = [qid for qid in sorted(qrels) if qid in queries]
    start = max(0, int(args.offset or 0))
    limit = _effective_limit(args, "scifact_retrieval")
    selected_query_ids = query_ids[start : start + max(1, limit)]
    selected_doc_ids: set[str] = set()
    if bool(args.index_full_corpus):
        selected_doc_ids = set(corpus.keys())
    else:
        for qid in selected_query_ids:
            selected_doc_ids.update(qrels.get(qid, set()))
        # Add deterministic nearby negatives so metrics are meaningful in mini mode.
        for doc_id in sorted(corpus.keys()):
            selected_doc_ids.add(doc_id)
            if len(selected_doc_ids) >= max(100, len(selected_query_ids) * 12):
                break

    collection = args.qdrant_collection or f"cf_eval_scifact_{paths.run_dir.name}".lower()
    collection = re.sub(r"[^a-z0-9_.-]+", "_", collection)
    client = QdrantClient(
        url=settings.qdrant_url,
        check_compatibility=False,
        trust_env=False,
        timeout=max(float(settings.qdrant_timeout_s), 20.0),
    )
    memory = QdrantMemory(client=client, embedding_model="BAAI/bge-m3", strict_schema=True)

    index_started = time.perf_counter()
    indexed = 0
    index_failures: list[str] = []
    for doc_id in sorted(selected_doc_ids):
        doc = corpus.get(doc_id)
        if not isinstance(doc, dict):
            continue
        content = " ".join(str(doc.get(k, "") or "").strip() for k in ("title", "text") if str(doc.get(k, "") or "").strip())
        if not content:
            continue
        vector = _encode_text(content)
        ok = await memory.upsert_memory(
            collection_name=collection,
            dense_vector=vector,
            point_id=str(uuid5(NAMESPACE_URL, f"{collection}:{doc_id}")),
            evidence=EvidenceItem(
                content=content,
                source_type="beir_scifact",
                source="BeIR/scifact",
                chunk_id=doc_id,
                title=str(doc.get("title", "") or ""),
                embedding_model="BAAI/bge-m3",
                metadata={"doc_id": doc_id, "stage": "scifact_retrieval"},
            ),
            wait=True,
        )
        if ok:
            indexed += 1
        else:
            index_failures.append(doc_id)
    if indexed == 0:
        return [
            {
                "case_id": "scifact_index",
                "stage": "scifact_retrieval",
                "elapsed_s": round(time.perf_counter() - index_started, 3),
                "ok": False,
                "validation": {
                    "ok": False,
                    "reason": "scifact_index_failed",
                    "details": {"index_failures": index_failures[:20], "collection": collection},
                },
            }
        ]

    rows: list[dict[str, Any]] = []
    for qid in selected_query_ids:
        started = time.perf_counter()
        query = str(queries[qid].get("text") or queries[qid].get("query") or "")
        relevant = qrels.get(qid, set())
        validation: dict[str, Any]
        try:
            qvec = _encode_text(query)
            hits = await memory.search(
                collection_name=collection,
                dense_vector=qvec,
                limit=10,
                min_score=-1.0,
            )
            ranked_ids = [str(item.chunk_id or (item.metadata or {}).get("doc_id", "")) for item in hits]
            metric = retrieval_metrics(relevant, ranked_ids)
            validation = {
                "ok": metric.get("recall_at_10", 0.0) > 0.0,
                "reason": "" if metric.get("recall_at_10", 0.0) > 0.0 else "no_relevant_hit_at_10",
                "details": {
                    "query": query,
                    "relevant_doc_ids": sorted(relevant),
                    "ranked_doc_ids": ranked_ids,
                    "collection": collection,
                    "indexed_docs": indexed,
                },
                **metric,
            }
        except Exception as exc:
            validation = {
                "ok": False,
                "reason": "scifact_search_failed",
                "details": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            }
        row = {
            "case_id": f"scifact:{qid}",
            "stage": "scifact_retrieval",
            "elapsed_s": round(time.perf_counter() - started, 3),
            "ok": bool(validation.get("ok", False)),
            "validation": validation,
        }
        case = {
            "case_id": row["case_id"],
            "query_id": qid,
            "query": query,
            "relevant_doc_ids": sorted(relevant),
        }
        case_dir = paths.cases_dir / _safe_case_dir_name(row["case_id"])
        _write_json(case_dir / "input.json", case)
        _write_json(case_dir / "validation.json", validation)
        _write_json(case_dir / "summary.json", row)
        _append_jsonl(paths.raw_predictions_path, {"case_id": row["case_id"], "input": case, "validation": validation})
        _append_jsonl(paths.latency_trace_path, {"case_id": row["case_id"], "elapsed_s": row["elapsed_s"]})
        rows.append(row)
        if bool(args.stop_on_failure) and not row["ok"] and str(validation.get("reason", "")) in STOP_REASONS:
            break
    _write_json(
        paths.run_dir / "scifact_index.json",
        {
            "collection": collection,
            "indexed_docs": indexed,
            "index_failures": index_failures,
            "index_elapsed_s": round(time.perf_counter() - index_started, 3),
            "corpus_path": str(corpus_path),
            "queries_path": str(queries_path),
            "qrels_path": str(qrels_path),
        },
    )
    return rows


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "public_dataset_eval.v1",
        "created_at": _utc_now(),
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(ROOT_DIR),
        "model_independent": True,
    }


def _build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "stage": args.stage,
        "cases_path_exists": bool(args.cases_path and Path(args.cases_path).exists()),
        "predictions_path_exists": bool(args.predictions_path and Path(args.predictions_path).exists()),
        "scifact_dir_exists": bool(args.scifact_dir and Path(args.scifact_dir).exists()),
    }
    warnings: list[str] = []
    if args.stage in {"humaneval_plus", "bfcl", "simpleqa"}:
        if not checks["cases_path_exists"]:
            warnings.append("--cases-path is required and must exist.")
        if not checks["predictions_path_exists"]:
            warnings.append("--predictions-path is required and must exist.")
    if args.stage == "scifact_retrieval" and not checks["scifact_dir_exists"]:
        warnings.append("--scifact-dir must point to a local BEIR SciFact directory.")
    return {
        "schema_version": "public_dataset_eval.preflight.v1",
        "generated_at": _utc_now(),
        "checks": checks,
        "warnings": warnings,
        "ok": not warnings,
    }


async def run_eval(args: argparse.Namespace) -> EvalPaths:
    stage = str(args.stage)
    run_name = args.run_name or _run_name(stage)
    paths = _make_paths(run_name)
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    preflight = _build_preflight(args)
    _write_json(paths.config_path, _build_config(args))
    _write_json(paths.preflight_path, preflight)
    _write_json(
        paths.subset_manifest_path,
        {
            "schema_version": "public_dataset_eval.v1",
            "stage": stage,
            "offset": int(args.offset or 0),
            "limit": _effective_limit(args, stage),
            "batch_size": int(args.batch_size or 0),
            "cases_path": str(args.cases_path or ""),
            "predictions_path": str(args.predictions_path or ""),
            "scifact_dir": str(args.scifact_dir or ""),
            "dataset_sources": {
                "humaneval_plus": "https://github.com/evalplus/evalplus",
                "bfcl": "https://gorilla.cs.berkeley.edu/leaderboard",
                "simpleqa": "https://github.com/openai/simple-evals",
                "scifact_retrieval": "https://huggingface.co/datasets/BeIR/scifact",
            },
        },
    )
    if not preflight["ok"]:
        _write_json(paths.metrics_path, summarize_rows(stage, [], selected_case_count=0, offset=int(args.offset or 0), batch_size=int(args.batch_size or 0), stopped_early=True))
        _write_json(paths.failures_path, [{"case_id": "preflight", "validation": {"reason": "preflight_failed", "details": preflight["warnings"]}}])
        paths.report_path.write_text(_build_report(run_name, json.loads(paths.metrics_path.read_text(encoding="utf-8")), json.loads(paths.failures_path.read_text(encoding="utf-8")), paths), encoding="utf-8")
        print(json.dumps({"run_dir": str(paths.run_dir), "preflight": preflight}, ensure_ascii=False, indent=2))
        return paths

    if stage == "scifact_retrieval":
        rows = await _score_scifact(args, paths)
        selected_count = len(rows)
    else:
        rows = await _score_prediction_stage(args, paths)
        selected_count = len(rows)

    stopped_early = bool(rows and args.stop_on_failure and not rows[-1].get("ok", False) and (rows[-1].get("validation") or {}).get("reason") in STOP_REASONS)
    failures = [row for row in rows if not row.get("ok")]
    metrics = summarize_rows(
        stage,
        rows,
        selected_case_count=selected_count,
        offset=int(args.offset or 0),
        batch_size=int(args.batch_size or 0),
        stopped_early=stopped_early,
    )
    _write_json(paths.metrics_path, metrics)
    _write_json(paths.failures_path, failures)
    paths.report_path.write_text(_build_report(run_name, metrics, failures, paths), encoding="utf-8")
    print(json.dumps({"run_dir": str(paths.run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))
    return paths


def write_example_files(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl = lambda p, rows: p.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_jsonl(
        target_dir / "humaneval_plus_cases.jsonl",
        [
            {
                "stage": "humaneval_plus",
                "case_id": "HumanEval/0",
                "entry_point": "add",
                "prompt": "def add(a, b):",
                "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
            }
        ],
    )
    _write_jsonl(target_dir / "humaneval_plus_predictions.jsonl", [{"case_id": "HumanEval/0", "completion": "def add(a, b):\n    return a + b"}])
    _write_jsonl(
        target_dir / "bfcl_cases.jsonl",
        [
            {
                "stage": "bfcl",
                "case_id": "bfcl_simple_0",
                "prompt": "Get weather for Paris in Celsius.",
                "functions": [{"name": "get_weather", "parameters": {"city": "string", "unit": "string"}}],
                "expected_tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}],
            }
        ],
    )
    _write_jsonl(target_dir / "bfcl_predictions.jsonl", [{"case_id": "bfcl_simple_0", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}]}])
    _write_jsonl(
        target_dir / "simpleqa_cases.jsonl",
        [
            {
                "stage": "simpleqa",
                "case_id": "simpleqa_0",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "aliases": ["Paris, France"],
            }
        ],
    )
    _write_jsonl(target_dir / "simpleqa_predictions.jsonl", [{"case_id": "simpleqa_0", "answer": "Paris"}])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-independent public dataset eval scoring.")
    parser.add_argument("--stage", choices=["humaneval_plus", "bfcl", "simpleqa", "scifact_retrieval"], default="humaneval_plus")
    parser.add_argument("--cases-path", default="", help="Normalized JSONL cases for HumanEval+/BFCL/SimpleQA.")
    parser.add_argument("--predictions-path", default="", help="Prediction JSONL produced by any model/system.")
    parser.add_argument("--scifact-dir", default="", help="Local BEIR SciFact directory with corpus.jsonl, queries.jsonl, qrels/test.tsv.")
    parser.add_argument("--qdrant-collection", default="", help="Optional Qdrant collection for SciFact retrieval.")
    parser.add_argument("--index-full-corpus", action="store_true", help="Index full SciFact corpus instead of a mini corpus slice.")
    parser.add_argument("--run-name", default="", help="Optional deterministic run directory name.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=5.0, help="Per HumanEval+ subprocess timeout.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--write-examples", default="", help="Write normalized example case/prediction files and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.write_examples:
        write_example_files(Path(args.write_examples))
        print(json.dumps({"examples_dir": str(Path(args.write_examples).resolve())}, ensure_ascii=False))
        return
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
