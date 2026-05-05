# coding=utf-8
"""Role agent selector for Phase 6A."""

from __future__ import annotations

from typing import Any


def _normalize_intent(task_profile: dict[str, Any]) -> str:
    intent = task_profile.get("intent", "general_qa")
    if hasattr(intent, "value"):
        return str(intent.value)
    return str(intent)


def select_agents_for_state(state: dict[str, Any]) -> list[str]:
    """Select role agents from structured state only (no model/storage calls)."""
    decision = state.get("workflow_decision", {})
    if not isinstance(decision, dict):
        decision = {}
    if bool(decision.get("use_quick_answer", False)):
        return []

    profile = state.get("task_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    intent = _normalize_intent(profile)

    requires_research = bool(profile.get("requires_research", False))
    results = state.get("research_results", [])
    evidence_count = len(results) if isinstance(results, list) else 0

    if intent in {"architecture_design", "planning"}:
        return ["architect", "synthesizer"]

    if intent in {"debugging", "code_review", "code_modification", "code_generation"}:
        if requires_research or evidence_count >= 5:
            return ["researcher", "coder", "synthesizer"]
        return ["coder", "synthesizer"]

    if intent in {"research_summary", "rag_answering"}:
        return ["researcher", "synthesizer"]

    if intent == "document_writing":
        return ["writer", "synthesizer"]

    if intent == "general_qa":
        return ["writer"]

    return ["synthesizer"]

