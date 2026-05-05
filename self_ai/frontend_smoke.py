# coding=utf-8
"""Frontend smoke helper for Agent Cockpit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from self_ai.frontend_app import run_frontend_workflow_once
from self_ai.frontend_formatting import (
    format_agent_turns,
    format_memory_context,
    format_quality_gate,
    format_review_reports,
    format_revision_summary,
    format_run_overview,
    sanitize_payload,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_cases() -> list[dict[str, Any]]:
    return [
        {"name": "quick_answer", "question": "Explain Redis role in one sentence."},
        {"name": "document_writing", "question": "Write a short project description for this multi-agent system."},
        {"name": "code_generation", "question": "Write a Python function that returns average of a list and handles empty list."},
        {"name": "architecture_design", "question": "Design the next-stage public benchmark evaluation plan for this project."},
        {"name": "research_summary", "question": "Summarize the roles of Redis, Qdrant, and Neo4j in this project."},
    ]


def _fake_result(case_name: str) -> dict[str, Any]:
    mode_map = {
        "quick_answer": "quick_answer",
        "document_writing": "document_writing",
        "code_generation": "code_focused",
        "architecture_design": "architecture_design",
        "research_summary": "research_then_implement",
    }
    selected = {
        "quick_answer": [],
        "document_writing": ["writer", "synthesizer"],
        "code_generation": ["coder", "synthesizer"],
        "architecture_design": ["architect", "synthesizer"],
        "research_summary": ["researcher", "synthesizer"],
    }.get(case_name, ["synthesizer"])
    now = _now_iso()
    return {
        "response": f"fake response for {case_name}",
        "model": "mock-model",
        "run_id": str(uuid4()),
        "session_id": "smoke-session",
        "task_profile": {"intent": case_name},
        "workflow_decision": {
            "execution_mode": mode_map.get(case_name, "general"),
            "use_quick_answer": case_name == "quick_answer",
        },
        "plan": {"steps": [{"step_id": "s1", "goal": "step"}]} if case_name != "quick_answer" else {},
        "context_pack": {
            "evidence_items": [{"source_type": "doc", "content": "summary", "score": 0.9}],
            "source_type_groups": {"doc": ["x"]},
            "context_preview": "short context",
            "graph_paths": [{"relation": "RELATED_TO"}],
            "graph_summary": {"relation_type_counts": {"RELATED_TO": 1}},
            "graph_context_preview": "graph preview",
            "token_budget": 2000,
        },
        "selected_agents": selected,
        "agent_outputs": {
            agent: {
                "agent_name": agent,
                "role": agent,
                "summary": f"{agent} summary",
                "output": f"{agent} output",
                "confidence": 0.8,
                "warnings": [],
                "used_context": ["context_pack"],
                "metadata": {},
            }
            for agent in selected
        },
        "review_reports": [] if case_name == "quick_answer" else [{"pass_review": True, "severity": "low"}],
        "quality_gate": {}
        if case_name == "quick_answer"
        else {"decision": "pass", "passed": True, "requires_revision": False, "severity": "low"},
        "errors": [],
        "created_at": now,
        "updated_at": now,
        "metadata": {
            "agent_runtime_used": case_name != "quick_answer",
            "revision_count": 0,
            "max_revision_iterations": 1,
            "revision_performed": False,
            "revision_target_agent": None,
            "pipeline": ["mainloop"],
            "elapsed_ms": 1234,
        },
    }


def _render_markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 7R Step 4A Agent Cockpit Smoke Report",
        "",
        f"- generated_at: {_now_iso()}",
        f"- case_count: {len(results)}",
        "",
        "| case | success | execution_mode | selected_agents | review_count | gate_decision | error |",
        "|---|---:|---|---|---:|---|---|",
    ]
    for row in results:
        lines.append(
            "| "
            + str(row.get("case", ""))
            + " | "
            + ("yes" if row.get("success") else "no")
            + " | "
            + str(row.get("execution_mode", ""))
            + " | "
            + ",".join(str(x) for x in row.get("selected_agents", []))
            + " | "
            + str(row.get("review_count", 0))
            + " | "
            + str(row.get("quality_gate_decision", ""))
            + " | "
            + str(row.get("error", ""))
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def run_frontend_smoke_cases(
    *,
    cases: list[dict] | None = None,
    fake_mode: bool = True,
    output_dir: str | Path = "docs/reports",
) -> list[dict]:
    case_items = cases or _default_cases()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in case_items:
        name = str(case.get("name", "case"))
        question = str(case.get("question", "")).strip()
        row: dict[str, Any] = {"case": name, "question": question, "success": False}

        try:
            if fake_mode:
                result = _fake_result(name)
                logs = [
                    '[self-ai] {"event":"engine.run.start","ts":"2026-01-01T00:00:00Z"}',
                    '[self-ai] {"event":"engine.pipeline.selected","ts":"2026-01-01T00:00:01Z","fields":{"pipeline_name":"%s"}}'
                    % result.get("workflow_decision", {}).get("execution_mode", "general"),
                ]
                error = None
            else:
                run_data = run_frontend_workflow_once(question)
                result = run_data.get("result")
                error = run_data.get("error")
                logs = run_data.get("logs", [])

            overview = format_run_overview(result, logs=logs)
            turns = format_agent_turns(result)
            reviews = format_review_reports(result)
            gate = format_quality_gate(result)
            revision = format_revision_summary(result, logs=logs)
            memory = format_memory_context(result)

            row.update(
                {
                    "success": error is None and bool(overview.get("run_id", "")),
                    "execution_mode": overview.get("execution_mode", ""),
                    "selected_agents": turns.get("selected_agents", []),
                    "plan_step_count": len((result or {}).get("plan", {}).get("steps", []))
                    if isinstance(result, dict)
                    else 0,
                    "review_count": reviews.get("count", 0),
                    "quality_gate_decision": gate.get("decision", ""),
                    "revision_count": revision.get("revision_count", 0),
                    "evidence_count": memory.get("evidence_count", 0),
                    "error": "" if error is None else str(_as_dict(error).get("message", "error")),
                }
            )
        except Exception as exc:
            row.update({"success": False, "error": f"{type(exc).__name__}:{exc}"})

        rows.append(row)

    json_path = out_dir / "phase7r_step4a_agent_cockpit_smoke.json"
    md_path = out_dir / "phase7r_step4a_agent_cockpit_smoke.md"
    json_path.write_text(_safe_json(rows), encoding="utf-8")
    md_path.write_text(_render_markdown_report(rows), encoding="utf-8")
    return rows


def _safe_json(value: Any) -> str:
    return json.dumps(sanitize_payload(value), ensure_ascii=False, indent=2, default=str)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
