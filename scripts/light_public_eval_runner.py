# coding=utf-8
"""Artifact-first lightweight public evaluation runner for Self AI.

This runner starts with the cheap runtime-smoke stage and uses the same artifact
layout planned for public benchmark subsets. It does not hide failures: every
case writes input, result, error, trace, validation, and summary files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from self_ai.config import resolve_project_path, settings  # noqa: E402
from self_ai.frontend_formatting import sanitize_payload  # noqa: E402
from self_ai.main import run_autonomy_workflow  # noqa: E402
from self_ai.observability import register_trace_listener, unregister_trace_listener  # noqa: E402

RUNS_ROOT = ROOT_DIR / "benchmarks" / "runs"

RUNTIME_SMOKE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "runtime.quick_answer.redis_role",
        "category": "runtime_smoke",
        "prompt": "In one concise sentence, explain Redis' role in this Self AI project.",
        "expect_substrings": ["redis"],
    },
    {
        "case_id": "runtime.project_summary",
        "category": "runtime_smoke",
        "prompt": "Write a short two-sentence project summary for Self AI.",
        "expect_substrings": ["self ai"],
    },
    {
        "case_id": "runtime.code_generation.average",
        "category": "runtime_smoke",
        "prompt": (
            "Return a Python function named average that computes the average "
            "of a list and raises ValueError for an empty list. Do not create files."
        ),
        "expect_substrings": ["def average", "ValueError"],
    },
    {
        "case_id": "runtime.release_eval_plan",
        "category": "runtime_smoke",
        "prompt": "Give three bullet points for evaluating a coding-agent project before a GitHub release.",
        "expect_substrings": ["release"],
        "min_bullets": 3,
    },
    {
        "case_id": "runtime.storage_boundaries",
        "category": "runtime_smoke",
        "prompt": (
            "Summarize the separate roles of Redis, Qdrant, and Neo4j in this "
            "project. Keep it under 90 words."
        ),
        "expect_substrings": ["redis", "qdrant", "neo4j"],
    },
]

PUBLIC_STAGE_MANIFESTS: list[dict[str, str]] = [
    {
        "stage": "code_correctness",
        "dataset": "EvalPlus HumanEval+ mini",
        "source_url": "https://github.com/evalplus/evalplus",
        "planned_subset": "first 20 stable HumanEval+ task ids",
        "status": "planned_not_run_by_default",
    },
    {
        "stage": "tool_calling",
        "dataset": "BFCL mini",
        "source_url": "https://gorilla.cs.berkeley.edu/leaderboard",
        "planned_subset": "50 single-turn non-live examples",
        "status": "planned_not_run_by_default",
    },
    {
        "stage": "factuality",
        "dataset": "SimpleQA mini",
        "source_url": "https://github.com/openai/simple-evals",
        "planned_subset": "100 fixed questions",
        "status": "planned_not_run_by_default",
    },
    {
        "stage": "retrieval_quality",
        "dataset": "BEIR SciFact mini",
        "source_url": "https://github.com/beir-cellar/beir",
        "planned_subset": "50 fixed SciFact queries",
        "status": "planned_not_run_by_default",
    },
]


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
    return f"{stamp}_{safe or 'light_public_eval'}"


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
        json.dumps(sanitize_payload(payload), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sanitize_payload(row), ensure_ascii=False, default=str) + "\n")


def _trace_counts(trace_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "trace_events": len(trace_rows),
        "model_calls": 0,
        "tool_call_end": 0,
        "tool_call_error": 0,
        "terminal_blocks": 0,
    }
    for row in trace_rows:
        event = str(row.get("event", ""))
        if event == "route_model.end":
            counts["model_calls"] += 1
        elif event == "tool.call.end":
            counts["tool_call_end"] += 1
        elif event == "tool.call.error":
            counts["tool_call_error"] += 1
        elif event in {"mainloop.final_answer.blocked", "mainloop.final_answer.terminal_block"}:
            counts["terminal_blocks"] += 1
    return counts


def _validate_case(case: dict[str, Any], result: dict[str, Any] | None, error: dict[str, Any] | None) -> dict[str, Any]:
    if error:
        return {"ok": False, "reason": "unhandled_exception", "details": error.get("message", "")}
    if not isinstance(result, dict):
        return {"ok": False, "reason": "missing_result", "details": "result is not a dict"}
    structured_errors = result.get("errors", [])
    if isinstance(structured_errors, list) and structured_errors:
        return {"ok": False, "reason": "structured_errors", "details": structured_errors[:3]}
    response = str(result.get("response", "") or "")
    if not response.strip():
        return {"ok": False, "reason": "empty_response", "details": ""}
    lowered = response.lower()
    min_bullets = int(case.get("min_bullets", 0) or 0)
    if min_bullets > 0:
        bullet_count = sum(
            1
            for line in response.splitlines()
            if line.lstrip().startswith(("-", "*", "•"))
        )
        if bullet_count < min_bullets:
            return {
                "ok": False,
                "reason": "min_bullets_not_met",
                "details": {"expected": min_bullets, "actual": bullet_count},
            }
    missing = [item for item in case.get("expect_substrings", []) if str(item).lower() not in lowered]
    if missing:
        return {"ok": False, "reason": "expected_substrings_missing", "details": missing}
    return {"ok": True, "reason": "", "details": ""}


async def _run_case(case: dict[str, Any], paths: EvalPaths) -> dict[str, Any]:
    case_id = str(case["case_id"])
    safe_case_id = case_id.replace("/", "_").replace("\\", "_")
    case_dir = paths.cases_dir / safe_case_id
    trace_rows: list[dict[str, Any]] = []

    def listener(payload: dict[str, Any]) -> None:
        trace_rows.append(dict(payload))

    started = time.perf_counter()
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    register_trace_listener(listener)
    try:
        result = await run_autonomy_workflow(
            str(case["prompt"]),
            session_id=f"eval-{safe_case_id}",
            chat_id=f"eval-{safe_case_id}",
        )
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        unregister_trace_listener(listener)
    elapsed_s = time.perf_counter() - started

    validation = _validate_case(case, result, error)
    counts = _trace_counts(trace_rows)
    summary = {
        "case_id": case_id,
        "category": case.get("category", ""),
        "started_at": _utc_now(),
        "elapsed_s": round(elapsed_s, 3),
        "ok": bool(validation.get("ok")),
        "validation": validation,
        "model": (result or {}).get("model", ""),
        "run_id": (result or {}).get("run_id", ""),
        "workflow_decision": (result or {}).get("workflow_decision", {}),
        "trace_counts": counts,
        "response_preview": str((result or {}).get("response", "") or "")[:500],
        "error": error,
    }

    _write_json(case_dir / "input.json", case)
    _write_json(case_dir / "result.json", result or {})
    _write_json(case_dir / "error.json", error or {})
    _write_json(case_dir / "validation.json", validation)
    _write_json(case_dir / "trace.json", trace_rows)
    _write_json(case_dir / "summary.json", summary)
    _append_jsonl(paths.raw_predictions_path, {"case_id": case_id, "input": case, "result": result or {}, "error": error})
    _append_jsonl(paths.latency_trace_path, {"case_id": case_id, "elapsed_s": round(elapsed_s, 3), **counts})
    return summary


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = (len(ordered) - 1) * pct
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def _failure_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("ok"):
            continue
        validation = row.get("validation", {}) if isinstance(row.get("validation"), dict) else {}
        reason = str(validation.get("reason", "unknown") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok_count = sum(1 for row in rows if bool(row.get("ok")))
    latencies = [float(row.get("elapsed_s", 0) or 0) for row in rows]
    model_calls = [int(row.get("trace_counts", {}).get("model_calls", 0) or 0) for row in rows]
    tool_calls = [int(row.get("trace_counts", {}).get("tool_call_end", 0) or 0) for row in rows]
    return {
        "schema_version": "light_public_eval.v1",
        "generated_at": _utc_now(),
        "total_cases": total,
        "ok_cases": ok_count,
        "failed_cases": total - ok_count,
        "workflow_ok_rate": round(ok_count / total, 4) if total else 0.0,
        "latency_avg_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "avg_model_calls": round(statistics.mean(model_calls), 3) if model_calls else 0.0,
        "avg_tool_calls": round(statistics.mean(tool_calls), 3) if tool_calls else 0.0,
        "failure_reasons": _failure_reasons(rows),
    }


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "light_public_eval.v1",
        "created_at": _utc_now(),
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(ROOT_DIR),
        "settings": {
            "router_default_tier": settings.router_default_tier,
            "router_use_selector": settings.router_use_selector,
            "model_fast": settings.model_fast,
            "model_balanced": settings.model_balanced,
            "model_high": settings.model_high,
            "model_code": settings.model_code,
            "permission_mode": settings.permission_mode,
            "chat_memory_enabled": settings.chat_memory_enabled,
            "chat_memory_fusion_enabled": settings.chat_memory_fusion_enabled,
            "redis_enabled": settings.redis_enabled,
            "qdrant_url": settings.qdrant_url,
            "neo4j_enabled": settings.neo4j_enabled,
            "graph_read_enabled": settings.graph_read_enabled,
        },
    }


def _build_preflight() -> dict[str, Any]:
    embedder_path = resolve_project_path(settings.embedder_path)
    checks = {
        "dashscope_api_key_loaded": bool(settings.dashscope_api_key),
        "redis_enabled": bool(settings.redis_enabled),
        "qdrant_url": settings.qdrant_url,
        "neo4j_enabled": bool(settings.neo4j_enabled),
        "embedder_prewarm": bool(settings.embedder_prewarm),
        "embedder_path": str(embedder_path),
        "embedder_path_exists": embedder_path.exists(),
        "chat_memory_enabled": bool(settings.chat_memory_enabled),
        "chat_memory_fusion_enabled": bool(settings.chat_memory_fusion_enabled),
    }
    warnings: list[str] = []
    if not checks["dashscope_api_key_loaded"]:
        warnings.append("DASHSCOPE_API_KEY is missing; model calls will fail.")
    if checks["embedder_prewarm"] and not checks["embedder_path_exists"]:
        warnings.append("SELF_AI_EMBEDDER_PREWARM=true but local embedder path is missing.")
    if settings.chat_memory_enabled and settings.chat_memory_fusion_enabled and not checks["embedder_path_exists"]:
        warnings.append("Chat memory L2 retrieval/write may fail without the local BGE-M3 model.")
    return {
        "schema_version": "light_public_eval.preflight.v1",
        "generated_at": _utc_now(),
        "checks": checks,
        "warnings": warnings,
        "ok": not warnings,
    }


def _build_report(run_name: str, metrics: dict[str, Any], failures: list[dict[str, Any]], paths: EvalPaths) -> str:
    lines = [
        "# Self AI Lightweight Public Eval Report",
        "",
        f"Run: `{run_name}`",
        f"Generated: `{metrics.get('generated_at', '')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total cases | {metrics.get('total_cases', 0)} |",
        f"| OK cases | {metrics.get('ok_cases', 0)} |",
        f"| Failed cases | {metrics.get('failed_cases', 0)} |",
        f"| Workflow OK rate | {metrics.get('workflow_ok_rate', 0)} |",
        f"| Latency avg (s) | {metrics.get('latency_avg_s', 0)} |",
        f"| Latency p50 (s) | {metrics.get('latency_p50_s', 0)} |",
        f"| Latency p95 (s) | {metrics.get('latency_p95_s', 0)} |",
        f"| Avg model calls | {metrics.get('avg_model_calls', 0)} |",
        f"| Avg tool calls | {metrics.get('avg_tool_calls', 0)} |",
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
    if not failures:
        lines.append("No failed cases.")
    else:
        lines.extend(["| Case | Reason | Details |", "|---|---|---|"])
        for row in failures:
            validation = row.get("validation", {}) if isinstance(row.get("validation"), dict) else {}
            details = str(validation.get("details", ""))[:160].replace("\n", " ")
            lines.append(f"| `{row.get('case_id', '')}` | `{validation.get('reason', '')}` | {details} |")
    lines.extend(
        [
            "",
            "## Release Note Guidance",
            "",
            "Use these numbers as a reproducibility smoke, not leaderboard-equivalent claims.",
            "Publish the run directory or attach it to the GitHub release so users can inspect raw outputs.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_eval(args: argparse.Namespace) -> EvalPaths:
    run_name = args.run_name or _run_name("light_public_eval")
    paths = _make_paths(run_name)
    paths.run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(paths.config_path, _build_config(args))
    _write_json(paths.preflight_path, _build_preflight())
    _write_json(
        paths.subset_manifest_path,
        {
            "schema_version": "light_public_eval.v1",
            "selected_stage": args.stage,
            "limit": args.limit,
            "runtime_smoke_cases": RUNTIME_SMOKE_CASES,
            "public_stage_manifests": PUBLIC_STAGE_MANIFESTS,
        },
    )

    start = max(0, int(args.offset or 0))
    selected_cases = RUNTIME_SMOKE_CASES[start:]
    if args.batch_size:
        selected_cases = selected_cases[: max(1, int(args.batch_size))]
    if args.limit:
        selected_cases = selected_cases[: max(1, int(args.limit))]

    rows: list[dict[str, Any]] = []
    stopped_early = False
    for case in selected_cases:
        row = await _run_case(case, paths)
        rows.append(row)
        if bool(args.stop_on_failure) and not bool(row.get("ok", False)):
            stopped_early = True
            break
    failures = [row for row in rows if not row.get("ok")]
    metrics = _summarize(rows)
    metrics["selected_case_count"] = len(selected_cases)
    metrics["executed_case_count"] = len(rows)
    metrics["offset"] = start
    metrics["batch_size"] = int(args.batch_size or 0)
    metrics["stopped_early"] = stopped_early
    _write_json(paths.metrics_path, metrics)
    _write_json(paths.failures_path, failures)
    paths.report_path.write_text(_build_report(run_name, metrics, failures, paths), encoding="utf-8")
    print(json.dumps({"run_dir": str(paths.run_dir), "metrics": metrics}, ensure_ascii=False, indent=2))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight public eval for Self AI.")
    parser.add_argument("--stage", default="runtime_smoke", choices=["runtime_smoke"])
    parser.add_argument("--run-name", default="", help="Optional deterministic run directory name.")
    parser.add_argument("--limit", type=int, default=0, help="Optional case limit for quick debugging.")
    parser.add_argument("--offset", type=int, default=0, help="Start case offset within the selected stage.")
    parser.add_argument("--batch-size", type=int, default=0, help="Maximum cases to run in this batch.")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the batch immediately after the first failed case.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
