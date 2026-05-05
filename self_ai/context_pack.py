# coding=utf-8
"""ContextPack models and builders for Phase 5B."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from .text_utils import shorten_text


class ContextPack(BaseModel):
    run_id: str = ""
    session_id: str = "default"
    task_summary: str = ""
    task_profile: dict[str, Any] = Field(default_factory=dict)
    workflow_decision: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    source_type_groups: dict[str, int] = Field(default_factory=dict)
    context_preview: str = ""
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    graph_summary: dict[str, Any] = Field(default_factory=dict)
    graph_context_preview: str = ""
    token_budget: int = 6000
    metadata: dict[str, Any] = Field(default_factory=dict)


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    return {}


def _compact_evidence(item: dict[str, Any]) -> dict[str, Any]:
    content = shorten_text(str(item.get("content", "")), max_chars=220)
    return {
        "content_preview": content,
        "source_type": str(item.get("source_type", "unknown")),
        "source": item.get("source"),
        "path": item.get("path"),
        "symbol": item.get("symbol"),
        "chunk_id": item.get("chunk_id"),
        "score": item.get("score"),
        "title": item.get("title"),
        "url": item.get("url"),
    }


def _extract_evidence(
    research_results: Any,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    if not isinstance(research_results, list):
        return [], {}, ""

    evidence_items: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    preview_lines: list[str] = []
    preview_budget = 600
    used = 0

    for raw in research_results:
        if not isinstance(raw, dict):
            continue
        compact = _compact_evidence(raw)
        evidence_items.append(compact)
        source_type = compact["source_type"]
        source_counts[source_type] += 1

        line = f"[{source_type}] {compact['content_preview']}".strip()
        pending = len(line) + 1
        if used + pending <= preview_budget:
            preview_lines.append(line)
            used += pending

    return evidence_items, dict(source_counts), "\n".join(preview_lines)


def _extract_graph_context(state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    # Preferred source: existing context_pack produced by retrieval/context builder.
    existing_cp = _to_dict(state.get("context_pack"))
    if existing_cp:
        graph_paths = existing_cp.get("graph_paths", [])
        graph_summary = _to_dict(existing_cp.get("graph_summary"))
        graph_preview = shorten_text(
            str(existing_cp.get("graph_context_preview", "")),
            max_chars=800,
        )
        if isinstance(graph_paths, list):
            compact_paths = []
            for path in graph_paths[:8]:
                if isinstance(path, dict):
                    compact_paths.append(
                        {
                            "nodes": path.get("nodes", []),
                            "edges": path.get("edges", []),
                            "metadata": _to_dict(path.get("metadata")),
                        }
                    )
            return compact_paths, graph_summary, graph_preview

    # Secondary source: state-level graph keys (if injected by future nodes).
    graph_paths = state.get("graph_paths", [])
    if not isinstance(graph_paths, list):
        graph_paths = []
    compact_paths: list[dict[str, Any]] = []
    for path in graph_paths[:8]:
        if isinstance(path, dict):
            compact_paths.append(
                {
                    "nodes": path.get("nodes", []),
                    "edges": path.get("edges", []),
                    "metadata": _to_dict(path.get("metadata")),
                }
            )
    graph_summary = _to_dict(state.get("graph_summary"))
    graph_preview = shorten_text(str(state.get("graph_context_preview", "")), max_chars=800)
    return compact_paths, graph_summary, graph_preview


def build_context_pack_from_state(
    state: dict[str, Any],
    *,
    token_budget: int = 6000,
) -> ContextPack:
    """Build a compact first-class ContextPack from workflow state."""
    run_id = str(state.get("run_id", ""))
    session_id = str(state.get("session_id", "default"))
    task_summary = shorten_text(str(state.get("task", "")), max_chars=240)
    task_profile = _to_dict(state.get("task_profile"))
    workflow_decision = _to_dict(state.get("workflow_decision"))
    plan = _to_dict(state.get("plan"))

    evidence_items, source_groups, context_preview = _extract_evidence(
        state.get("research_results", [])
    )
    graph_paths, graph_summary, graph_context_preview = _extract_graph_context(state)

    return ContextPack(
        run_id=run_id,
        session_id=session_id,
        task_summary=task_summary,
        task_profile=task_profile,
        workflow_decision=workflow_decision,
        plan=plan,
        evidence_items=evidence_items,
        source_type_groups=source_groups,
        context_preview=shorten_text(context_preview, max_chars=900),
        graph_paths=graph_paths,
        graph_summary=graph_summary,
        graph_context_preview=shorten_text(graph_context_preview, max_chars=900),
        token_budget=max(800, int(token_budget)),
        metadata={
            "evidence_count": len(evidence_items),
            "graph_path_count": len(graph_paths),
            "has_plan": bool(plan),
        },
    )


def context_pack_to_prompt_block(
    context_pack: ContextPack | dict[str, Any] | None,
    *,
    max_chars: int = 6000,
) -> str:
    """Render compact prompt block from ContextPack."""
    if context_pack is None:
        return ""
    cp = context_pack if isinstance(context_pack, ContextPack) else ContextPack(**_to_dict(context_pack))

    sections: list[str] = []

    if cp.task_summary:
        sections.append(f"Task Summary:\n{shorten_text(cp.task_summary, max_chars=320)}")

    if cp.workflow_decision:
        wd = cp.workflow_decision
        sections.append(
            "Workflow Decision:\n"
            f"- execution_mode: {wd.get('execution_mode')}\n"
            f"- next_node: {wd.get('next_node')}\n"
            f"- skip_research: {wd.get('skip_research')}\n"
            f"- skip_debate: {wd.get('skip_debate')}\n"
            f"- requires_context: {wd.get('requires_context')}"
        )

    plan_steps = _to_dict(cp.plan).get("steps", [])
    if isinstance(plan_steps, list) and plan_steps:
        lines = ["Plan:"]
        for raw in plan_steps[:6]:
            if not isinstance(raw, dict):
                continue
            sid = raw.get("step_id", "?")
            goal = shorten_text(str(raw.get("goal", "")), max_chars=160)
            done = shorten_text(str(raw.get("done_criteria", "")), max_chars=120)
            lines.append(f"- {sid}: {goal}")
            if done:
                lines.append(f"  done: {done}")
        sections.append("\n".join(lines))

    if cp.evidence_items:
        lines = ["Semantic Evidence Summary:"]
        if cp.source_type_groups:
            lines.append(f"- source_type_counts: {cp.source_type_groups}")
        for item in cp.evidence_items[:5]:
            lines.append(
                f"- [{item.get('source_type', 'unknown')}] "
                f"{shorten_text(str(item.get('content_preview', '')), max_chars=160)}"
            )
        sections.append("\n".join(lines))

    if cp.graph_summary or cp.graph_context_preview:
        lines = ["Graph Context Summary:"]
        if cp.graph_summary:
            lines.append(f"- graph_summary: {cp.graph_summary}")
        if cp.graph_context_preview:
            lines.append(
                f"- preview: {shorten_text(cp.graph_context_preview, max_chars=280)}"
            )
        if cp.graph_paths:
            lines.append(f"- path_count: {len(cp.graph_paths)}")
        sections.append("\n".join(lines))

    block = "\n\n".join(section for section in sections if section.strip())
    return shorten_text(block, max_chars=max(800, int(max_chars)))
