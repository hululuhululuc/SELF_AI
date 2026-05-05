# coding=utf-8
"""Pipeline stage definitions for EngineLoop."""

from __future__ import annotations

from typing import Any


PIPELINES: dict[str, list[str]] = {
    "quick_answer": ["quick_answer", "finalize"],
    "document_writing": ["plan", "context_pack", "agent_runtime", "final_response", "finalize"],
    "code_generation": ["research", "plan", "context_pack", "agent_runtime", "final_response", "finalize"],
    "debugging": ["research", "plan", "context_pack", "agent_runtime", "final_response", "finalize"],
    "architecture_design": ["research", "plan", "context_pack", "agent_runtime", "final_response", "finalize"],
    "research_summary": ["research", "plan", "context_pack", "agent_runtime", "final_response", "finalize"],
    "general": ["research", "plan", "context_pack", "agent_runtime", "final_response", "finalize"],
}


def _as_str(value: Any, default: str = "general") -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        return str(value.value)
    text = str(value).strip()
    return text or default


def select_pipeline(
    workflow_decision: dict[str, Any] | None,
    task_profile: dict[str, Any] | None = None,
) -> list[str]:
    """Select a stage list based on workflow decision/task profile."""
    decision = workflow_decision or {}
    profile = task_profile or {}

    if bool(decision.get("use_quick_answer", False)):
        return list(PIPELINES["quick_answer"])

    mode = _as_str(decision.get("execution_mode"), "general")
    intent = _as_str(profile.get("intent"), "general_qa")

    if mode == "document_writing":
        stages = list(PIPELINES["document_writing"])
    elif mode in {"code_focused"}:
        if intent in {"debugging", "code_review", "code_modification"}:
            stages = list(PIPELINES["debugging"])
        else:
            stages = list(PIPELINES["code_generation"])
    elif mode == "architecture_design":
        stages = list(PIPELINES["architecture_design"])
    elif mode == "research_then_implement":
        stages = list(PIPELINES["research_summary"])
    else:
        stages = list(PIPELINES["general"])

    if bool(decision.get("skip_research", False)) and "research" in stages:
        stages = [s for s in stages if s != "research"]

    return stages
