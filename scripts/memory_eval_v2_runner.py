"""Memory effectiveness real-eval runner (fresh run, v2).

This script always writes new artifacts under:
  self_ai/docs/memory_eval_v2/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from self_ai.config import settings
from self_ai.main import run_autonomy_workflow


ROOT_DIR = Path(__file__).resolve().parents[1]
if Path.cwd().resolve() != ROOT_DIR:
    os.chdir(ROOT_DIR)
# Runtime tools operate under `self_ai/` as project_root in production entrypoint.
RUNTIME_WORKSPACE_ROOT = ROOT_DIR / "self_ai"

BASE_DIR = ROOT_DIR / "docs/memory_eval_v2"
RAW_DIR = BASE_DIR / "raw"
SUMMARY_DIR = BASE_DIR / "summary"
METRICS_JSON = SUMMARY_DIR / "metrics.json"
REPORT_MD = SUMMARY_DIR / "report.md"
FAILURES_JSON = SUMMARY_DIR / "failures.json"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _chat_memory_root() -> Path:
    root = Path(settings.chat_memory_root)
    if not root.is_absolute():
        root = (ROOT_DIR / root).resolve()
    return root


def _contains_quota_error(payload: Any) -> bool:
    markers = (
        "insufficient_quota",
        "exceeded today's quota",
        "rate limit",
        "error code: 429",
        "quota",
        "tokens already used",
        "额度",
        "配额",
    )

    def _walk(v: Any) -> bool:
        if isinstance(v, dict):
            return any(_walk(x) for x in v.values())
        if isinstance(v, list):
            return any(_walk(x) for x in v)
        text = _safe_text(v).lower()
        return any(m in text for m in markers)

    return _walk(payload)


def _tool_calls_from_trace(trace_rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in trace_rows if _safe_text(row.get("event")) == "tool.call.end")


def _validate_case_a(rows: list[dict[str, Any]], workspace: Path) -> tuple[bool, str]:
    _ = workspace
    if not rows:
        return False, "no_turns"
    last = _safe_text(rows[-1].get("response")).lower()
    if "pref-ax7" not in last:
        return False, "preference_token_missing"
    return True, ""


def _validate_case_b(rows: list[dict[str, Any]], workspace: Path) -> tuple[bool, str]:
    final_file = workspace / "b_final_note.txt"
    if not final_file.exists():
        return False, "final_file_missing"
    content = final_file.read_text(encoding="utf-8")
    expected = "line1: hello\nline2: memory-proof\nline3: done\n"
    if content != expected:
        return False, "final_content_mismatch"
    if (workspace / "b_note.txt").exists():
        return False, "old_file_exists"
    return True, ""


def _validate_case_c(rows: list[dict[str, Any]], workspace: Path) -> tuple[bool, str]:
    f = workspace / "c_issue_fix.txt"
    if not f.exists():
        return False, "fix_file_missing"
    if "fixed" not in f.read_text(encoding="utf-8").lower():
        return False, "fix_content_invalid"
    last = _safe_text(rows[-1].get("response")).lower()
    if "error" not in last and "failed" not in last and "failure" not in last:
        return False, "failure_recap_missing"
    return True, ""


def _validate_case_d(rows: list[dict[str, Any]], workspace: Path) -> tuple[bool, str]:
    _ = workspace
    if not rows:
        return False, "no_turns"
    last = _safe_text(rows[-1].get("response")).lower()
    for key in ("redis", "qdrant", "neo4j"):
        if key not in last:
            return False, f"missing_{key}"
    return True, ""


def _validate_case_e(rows: list[dict[str, Any]], workspace: Path) -> tuple[bool, str]:
    _ = workspace
    if not rows:
        return False, "no_turns"
    chat_id = _safe_text(rows[0].get("chat_id"))
    manifest = _chat_memory_root() / chat_id / "manifest.json"
    if not manifest.exists():
        return False, "manifest_missing"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    archived = _as_dict(payload.get("archived_shards"))
    versions = _as_list(payload.get("summary_versions"))
    if not archived or not versions:
        return False, "compaction_not_observed"
    last = _safe_text(rows[-1].get("response")).lower()
    if "e-memory-lock-2026" not in last:
        return False, "cold_recall_missing"
    return True, ""


Validator = Callable[[list[dict[str, Any]], Path], tuple[bool, str]]


@dataclass
class TaskCase:
    code: str
    name: str
    turns: list[str]
    validator: Validator


def _cases() -> list[TaskCase]:
    return [
        TaskCase(
            code="A",
            name="preference_persistence",
            turns=[
                "Remember this user preference token: PREF-AX7. Keep future answers concise and bullet-oriented.",
                "Explain what this project does.",
                "Answer again using exactly 3 concise bullet points.",
                "What preference token did I set earlier? Return token and one-line explanation.",
            ],
            validator=_validate_case_a,
        ),
        TaskCase(
            code="B",
            name="file_state_consistency",
            turns=[
                "Create artifacts/memory_eval_v2_workspace/b_note.txt with exactly three lines: line1: hello, line2: memory-proof, line3: done.",
                "Ensure b_note.txt still has exactly those three lines and no extra lines.",
                "Rename artifacts/memory_eval_v2_workspace/b_note.txt to artifacts/memory_eval_v2_workspace/b_final_note.txt.",
                "Return final file path and repeat each line verbatim.",
            ],
            validator=_validate_case_b,
        ),
        TaskCase(
            code="C",
            name="issue_fix_reuse",
            turns=[
                "Read artifacts/memory_eval_v2_workspace/no_such_file.txt and report the exact error.",
                "Fix it by creating artifacts/memory_eval_v2_workspace/c_issue_fix.txt with text: fixed by runtime.",
                "Read c_issue_fix.txt and return its content.",
                "Summarize: what failed, what was changed, and how to avoid this next time.",
            ],
            validator=_validate_case_c,
        ),
        TaskCase(
            code="D",
            name="research_summary_continuity",
            turns=[
                "Summarize Redis, Qdrant, and Neo4j responsibility boundaries in this project.",
                "Rewrite as 3 lines focused on runtime state, semantic retrieval, and structural graph.",
                "Add one risk and one mitigation for each layer.",
                "Provide a final compact structured summary.",
            ],
            validator=_validate_case_d,
        ),
        TaskCase(
            code="E",
            name="compaction_cold_recall",
            turns=[
                "Remember this early constraint token: E-MEMORY-LOCK-2026.",
                "Create artifacts/memory_eval_v2_workspace/e1.txt with content e1.",
                "Create artifacts/memory_eval_v2_workspace/e2.txt with content e2.",
                "Create artifacts/memory_eval_v2_workspace/e3.txt with content e3.",
                "Create artifacts/memory_eval_v2_workspace/e4.txt with content e4.",
                "List the file actions you completed so far in one line.",
                "Repeat the early constraint token only, then one short explanation.",
                "After memory compaction, what was the earliest key constraint token?",
            ],
            validator=_validate_case_e,
        ),
    ]


def _prepare_dirs() -> None:
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR, ignore_errors=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_workspace() -> Path:
    ws = RUNTIME_WORKSPACE_ROOT / "artifacts/memory_eval_v2_workspace"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _cleanup_chat_memory(prefix: str) -> None:
    root = _chat_memory_root()
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            shutil.rmtree(child, ignore_errors=True)


async def _run_turn(case: TaskCase, mode: str, chat_id: str, idx: int, text: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = await run_autonomy_workflow(
        text,
        session_id=chat_id,
        chat_id=chat_id,
        user_id="memory-eval-v2",
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    trace_rows = [r for r in _as_list(result.get("trace")) if isinstance(r, dict)]
    metadata = _as_dict(result.get("metadata"))
    stats = _as_dict(metadata.get("chat_recent_turns_stats"))
    execution_state = _as_dict(metadata.get("execution_state"))
    control_events = _as_list(metadata.get("control_events"))
    row = {
        "timestamp": _now_utc(),
        "case": case.code,
        "case_name": case.name,
        "mode": mode,
        "chat_id": chat_id,
        "turn_index": idx,
        "task_text": text,
        "run_id": _safe_text(result.get("run_id")),
        "elapsed_ms": elapsed_ms,
        "tool_calls": _tool_calls_from_trace(trace_rows),
        "error_count": len(_as_list(result.get("errors"))),
        "errors": _as_list(result.get("errors")),
        "control_events": control_events,
        "control_event_count": len(control_events),
        "terminal_error_count": _safe_int(execution_state.get("terminal_error_count"), len(_as_list(result.get("errors")))),
        "response": _safe_text(result.get("response")),
        "workflow_decision": _as_dict(result.get("workflow_decision")),
        "source_type_groups": _as_dict(stats.get("source_type_groups")),
        "fusion_failed": bool(stats.get("fusion_failed", False)),
        "degraded": bool(stats.get("degraded", False)),
        "degraded_reasons": _as_list(stats.get("degraded_reasons")),
        "layer_health": _as_dict(stats.get("layer_health")),
        "result": result,
    }
    out_dir = RAW_DIR / chat_id / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"turn_{idx:03d}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return row


def _case_summary(
    case: TaskCase,
    mode: str,
    chat_id: str,
    rows: list[dict[str, Any]],
    validation_ok: bool,
    validation_reason: str,
    quota_exhausted: bool,
) -> dict[str, Any]:
    turns = len(rows)
    avg_ms = round(sum(float(r.get("elapsed_ms", 0.0)) for r in rows) / turns, 2) if turns else 0.0
    avg_tools = round(sum(float(r.get("tool_calls", 0.0)) for r in rows) / turns, 2) if turns else 0.0
    total_errors = sum(_safe_int(r.get("error_count"), 0) for r in rows)
    l1 = l2 = l3 = 0
    for r in rows:
        g = _as_dict(r.get("source_type_groups"))
        l1 += _safe_int(g.get("L1"), 0)
        l2 += _safe_int(g.get("L2"), 0)
        l3 += _safe_int(g.get("L3"), 0)
    return {
        "case": case.code,
        "case_name": case.name,
        "mode": mode,
        "chat_id": chat_id,
        "turn_count": turns,
        "validation_ok": validation_ok,
        "validation_reason": validation_reason,
        "quota_exhausted": quota_exhausted,
        "avg_elapsed_ms": avg_ms,
        "avg_tool_calls": avg_tools,
        "total_errors": total_errors,
        "L1_hits": l1,
        "L2_hits": l2,
        "L3_hits": l3,
    }


def _compare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(_safe_text(r.get("case")), {})[_safe_text(r.get("mode"))] = r
    out: list[dict[str, Any]] = []
    for case, mode_map in by_case.items():
        on = _as_dict(mode_map.get("ON"))
        off = _as_dict(mode_map.get("OFF"))
        out.append(
            {
                "case": case,
                "on_validation_ok": bool(on.get("validation_ok", False)),
                "off_validation_ok": bool(off.get("validation_ok", False)),
                "on_avg_elapsed_ms": on.get("avg_elapsed_ms"),
                "off_avg_elapsed_ms": off.get("avg_elapsed_ms"),
                "on_avg_tool_calls": on.get("avg_tool_calls"),
                "off_avg_tool_calls": off.get("avg_tool_calls"),
                "on_hits": {"L1": on.get("L1_hits"), "L2": on.get("L2_hits"), "L3": on.get("L3_hits")},
                "off_hits": {"L1": off.get("L1_hits"), "L2": off.get("L2_hits"), "L3": off.get("L3_hits")},
            }
        )
    return out


def _write_reports(payload: dict[str, Any]) -> None:
    METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    FAILURES_JSON.write_text(
        json.dumps(_as_list(payload.get("failures")), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Memory Eval V2 Report",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- stop_reason: `{payload.get('stop_reason', '')}`",
        "",
        "## Case Summary",
        "",
        "| Case | Mode | Valid | Reason | Turns | Avg ms | Avg tools | Errors | L1/L2/L3 | Quota |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in _as_list(payload.get("case_summaries")):
        if not isinstance(r, dict):
            continue
        lines.append(
            "| {case} {name} | {mode} | {ok} | {reason} | {turns} | {ms} | {tools} | {errs} | {l1}/{l2}/{l3} | {quota} |".format(
                case=r.get("case", ""),
                name=r.get("case_name", ""),
                mode=r.get("mode", ""),
                ok=r.get("validation_ok", False),
                reason=_safe_text(r.get("validation_reason", ""))[:40],
                turns=r.get("turn_count", 0),
                ms=r.get("avg_elapsed_ms", 0),
                tools=r.get("avg_tool_calls", 0),
                errs=r.get("total_errors", 0),
                l1=r.get("L1_hits", 0),
                l2=r.get("L2_hits", 0),
                l3=r.get("L3_hits", 0),
                quota=r.get("quota_exhausted", False),
            )
        )
    lines.extend(
        [
            "",
            "## ON vs OFF",
            "",
            "| Case | ON valid | OFF valid | ON ms | OFF ms | ON tools | OFF tools | ON L2 | OFF L2 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in _as_list(payload.get("compare_rows")):
        if not isinstance(r, dict):
            continue
        on_hits = _as_dict(r.get("on_hits"))
        off_hits = _as_dict(r.get("off_hits"))
        lines.append(
            "| {case} | {onv} | {offv} | {onms} | {offms} | {ont} | {offt} | {onl2} | {offl2} |".format(
                case=r.get("case", ""),
                onv=r.get("on_validation_ok", False),
                offv=r.get("off_validation_ok", False),
                onms=r.get("on_avg_elapsed_ms", "n/a"),
                offms=r.get("off_avg_elapsed_ms", "n/a"),
                ont=r.get("on_avg_tool_calls", "n/a"),
                offt=r.get("off_avg_tool_calls", "n/a"),
                onl2=on_hits.get("L2", 0),
                offl2=off_hits.get("L2", 0),
            )
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


async def run_eval(
    *,
    modes: list[str],
    chat_prefix: str,
    selected_cases: set[str] | None,
    stop_on_validation_fail: bool,
) -> dict[str, Any]:
    _prepare_dirs()
    workspace = _prepare_workspace()
    _cleanup_chat_memory(chat_prefix)

    # Keep compaction observable in a bounded turn count.
    original_shard_turns = int(settings.chat_memory_max_turns_per_shard)
    original_compact_min = int(settings.chat_memory_compact_min_shards)
    settings.chat_memory_max_turns_per_shard = max(10, min(original_shard_turns, 10))
    settings.chat_memory_compact_min_shards = max(2, min(original_compact_min, 2))

    failures: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    stop_reason = "completed"

    try:
        for case in _cases():
            if selected_cases and case.code.upper() not in selected_cases:
                continue
            for mode in modes:
                settings.chat_memory_fusion_enabled = mode == "ON"
                chat_id = f"{chat_prefix}-{case.code.lower()}-{mode.lower()}"
                rows: list[dict[str, Any]] = []
                quota_exhausted = False

                for idx, text in enumerate(case.turns, start=1):
                    row = await _run_turn(case, mode, chat_id, idx, text)
                    rows.append(row)
                    if _contains_quota_error(row):
                        quota_exhausted = True
                        failures.append(
                            {
                                "case": case.code,
                                "mode": mode,
                                "chat_id": chat_id,
                                "turn_index": idx,
                                "type": "QuotaExhausted",
                                "message": "Detected model quota/rate-limit exhaustion.",
                            }
                        )
                        break

                if quota_exhausted:
                    valid, reason = False, "quota_exhausted"
                else:
                    valid, reason = case.validator(rows, workspace)

                summary = _case_summary(case, mode, chat_id, rows, valid, reason, quota_exhausted)
                case_summaries.append(summary)

                if not valid:
                    failures.append(
                        {
                            "case": case.code,
                            "mode": mode,
                            "chat_id": chat_id,
                            "type": "ValidationFailed",
                            "reason": reason,
                        }
                    )

                if quota_exhausted:
                    stop_reason = f"stopped_on_quota:{case.code}:{mode}"
                    return {
                        "generated_at": _now_utc(),
                        "stop_reason": stop_reason,
                        "case_summaries": case_summaries,
                        "compare_rows": _compare_rows(case_summaries),
                        "failures": failures,
                    }

                if stop_on_validation_fail and not valid:
                    stop_reason = f"stopped_on_validation:{case.code}:{mode}:{reason}"
                    return {
                        "generated_at": _now_utc(),
                        "stop_reason": stop_reason,
                        "case_summaries": case_summaries,
                        "compare_rows": _compare_rows(case_summaries),
                        "failures": failures,
                    }
    finally:
        settings.chat_memory_max_turns_per_shard = original_shard_turns
        settings.chat_memory_compact_min_shards = original_compact_min

    return {
        "generated_at": _now_utc(),
        "stop_reason": stop_reason,
        "case_summaries": case_summaries,
        "compare_rows": _compare_rows(case_summaries),
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fresh memory eval v2.")
    parser.add_argument("--modes", default="ON,OFF", help="ON,OFF or ON or OFF.")
    parser.add_argument("--cases", default="A,B,C,D,E", help="Comma separated case codes.")
    parser.add_argument("--chat-prefix", default="m2eval", help="Chat ID prefix.")
    parser.add_argument(
        "--continue-on-validation-fail",
        action="store_true",
        help="Continue remaining tasks even when one validation fails.",
    )
    return parser.parse_args()


async def _amain() -> None:
    args = _parse_args()
    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    modes = [m for m in modes if m in {"ON", "OFF"}]
    if not modes:
        raise SystemExit("No valid mode. Use ON and/or OFF.")
    selected_cases = {c.strip().upper() for c in args.cases.split(",") if c.strip()}
    payload = await run_eval(
        modes=modes,
        chat_prefix=args.chat_prefix,
        selected_cases=selected_cases,
        stop_on_validation_fail=not bool(args.continue_on_validation_fail),
    )
    _write_reports(payload)
    print(
        json.dumps(
            {
                "generated_at": payload.get("generated_at"),
                "stop_reason": payload.get("stop_reason"),
                "case_count": len(_as_list(payload.get("case_summaries"))),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_amain())
