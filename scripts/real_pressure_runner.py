"""Real pressure runner for create/edit/rename/delete/retrieval/code-rewrite."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from self_ai.main import run_autonomy_workflow


CASES: list[dict[str, str]] = [
    {
        "id": "create",
        "chat_id": "real-pressure-fix-create",
        "task": "在项目目录 artifacts/real_pressure 下创建 t_create.txt，写入一行: hello pressure，并返回文件完整相对路径。",
    },
    {
        "id": "edit",
        "chat_id": "real-pressure-fix-edit",
        "task": "编辑 artifacts/real_pressure/t_create.txt，追加第二行: edit ok，并返回路径。",
    },
    {
        "id": "rename",
        "chat_id": "real-pressure-fix-rename",
        "task": "把 artifacts/real_pressure/t_create.txt 重命名为 artifacts/real_pressure/t_renamed.txt，并返回新路径。",
    },
    {
        "id": "delete",
        "chat_id": "real-pressure-fix-delete",
        "task": "删除 artifacts/real_pressure/t_renamed.txt，并确认该文件已不存在。",
    },
    {
        "id": "retrieval",
        "chat_id": "real-pressure-fix-retrieval",
        "task": "总结当前项目里 Redis、Qdrant、Neo4j 的边界职责，每个最多两句。",
    },
    {
        "id": "code_rewrite",
        "chat_id": "real-pressure-fix-rewrite",
        "task": "在 artifacts/real_pressure 下创建 t_rewrite.py，实现一个带注释的 quicksort，并返回路径。",
    },
]


def _is_blocked(result: dict[str, Any]) -> bool:
    response = str(result.get("response", "") or "")
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    if response.startswith("Execution not completed with verifiable evidence"):
        return True
    if bool(metadata.get("completion_gate_terminal", False)):
        return True
    blocked_count = int(metadata.get("completion_gate_blocked_count", 0) or 0)
    return blocked_count > 0 and not response.strip()


def _count_trace_events(trace: list[dict[str, Any]], event_name: str) -> int:
    count = 0
    for item in trace:
        if not isinstance(item, dict):
            continue
        if str(item.get("event", "")).strip() == event_name:
            count += 1
    return count


def _collect_case_metrics(case_id: str, result: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    trace = result.get("trace", []) if isinstance(result.get("trace"), list) else []
    recent_tool_results = metadata.get("recent_tool_results", [])
    tool_calls = _count_trace_events(trace, "tool.call.end")
    if tool_calls <= 0 and isinstance(recent_tool_results, list):
        tool_calls = len([x for x in recent_tool_results if isinstance(x, dict)])
    turn_count = _count_trace_events(trace, "loop.turn.start")

    chat_memory = metadata.get("chat_memory", {}) if isinstance(metadata.get("chat_memory"), dict) else {}
    sidecar = metadata.get("chat_memory_sidecar", {}) if isinstance(metadata.get("chat_memory_sidecar"), dict) else {}

    return {
        "id": case_id,
        "ok": True,
        "elapsed_s": round(elapsed_s, 2),
        "run_id": str(result.get("run_id", "") or ""),
        "model": str(result.get("model", "") or ""),
        "response_preview": str(result.get("response", "") or "")[:280],
        "errors_count": len(result.get("errors", [])) if isinstance(result.get("errors"), list) else 0,
        "blocked": _is_blocked(result),
        "turn_count": turn_count,
        "tool_calls": tool_calls,
        "selected_agents": result.get("selected_agents", []) if isinstance(result.get("selected_agents"), list) else [],
        "quality_gate": result.get("quality_gate", {}) if isinstance(result.get("quality_gate"), dict) else {},
        "memory_sidecar_used": bool(sidecar.get("succeeded", False)),
        "memory_sidecar_attempted": bool(sidecar.get("attempted", False)),
        "memory_committed_count": int(chat_memory.get("committed_memory_count", 0) or 0),
        "l2l3_failed": bool(metadata.get("chat_memory_l2l3_failed", False)),
        "l2l3_written": bool((metadata.get("chat_memory_l2l3_write", {}) or {}).get("qdrant_ok", 0) or (metadata.get("chat_memory_l2l3_write", {}) or {}).get("neo4j_ok", 0)),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total_cases": 0,
            "ok_cases": 0,
            "blocked_rate_pct": 0.0,
            "avg_turns": 0.0,
            "avg_tool_calls": 0.0,
            "avg_elapsed_s": 0.0,
        }
    blocked = sum(1 for row in rows if bool(row.get("blocked", False)))
    return {
        "total_cases": total,
        "ok_cases": sum(1 for row in rows if bool(row.get("ok", False))),
        "blocked_rate_pct": round((blocked / total) * 100.0, 2),
        "avg_turns": round(sum(float(row.get("turn_count", 0) or 0) for row in rows) / total, 2),
        "avg_tool_calls": round(sum(float(row.get("tool_calls", 0) or 0) for row in rows) / total, 2),
        "avg_elapsed_s": round(sum(float(row.get("elapsed_s", 0.0) or 0.0) for row in rows) / total, 2),
    }


async def _run_all() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        t0 = time.perf_counter()
        result = await run_autonomy_workflow(case["task"], session_id=case["chat_id"], chat_id=case["chat_id"])
        elapsed = time.perf_counter() - t0
        rows.append(_collect_case_metrics(case["id"], result, elapsed))
    return {"summary": _summarize(rows), "cases": rows}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_compare(before: dict[str, Any] | None, after: dict[str, Any]) -> dict[str, Any]:
    def _s(src: dict[str, Any] | None, key: str) -> float:
        summary = (src or {}).get("summary", {}) if isinstance((src or {}).get("summary", {}), dict) else {}
        try:
            return float(summary.get(key, 0.0) or 0.0)
        except Exception:
            return 0.0

    keys = ["blocked_rate_pct", "avg_turns", "avg_tool_calls", "avg_elapsed_s"]
    deltas = {key: round(_s(after, key) - _s(before, key), 2) for key in keys}
    return {
        "before_summary": (before or {}).get("summary", {}),
        "after_summary": after.get("summary", {}),
        "delta": deltas,
    }


def main() -> None:
    report_dir = Path("docs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    before_path = report_dir / "real_pressure_run_report.json"
    after_path = report_dir / "real_pressure_run_report_after_fix.json"
    compare_path = report_dir / "real_pressure_compare_after_fix.json"

    before = _load_json(before_path)
    after = asyncio.run(_run_all())
    compare = _build_compare(before, after)

    after_path.write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
    compare_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(compare, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

