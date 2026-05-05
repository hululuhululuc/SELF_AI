# coding=utf-8
"""Policy rules for AgentRuntime selection and revision behavior."""

from __future__ import annotations

from typing import Any

from .agent_session import AgentSession


class AgentPolicy:
    """Deterministic policy; no model/storage calls."""

    def _intent(self, session: AgentSession) -> str:
        value = session.task_profile.get("intent", "general_qa")
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)

    def select_agents(self, session: AgentSession) -> list[str]:
        if bool(session.workflow_decision.get("use_quick_answer", False)):
            return []
        intent = self._intent(session)
        requires_research = bool(session.task_profile.get("requires_research", False))
        evidence_count = (
            len(session.context_pack.get("evidence_items", []))
            if isinstance(session.context_pack.get("evidence_items"), list)
            else 0
        )

        if intent in {"architecture_design", "planning"}:
            return ["architect", "synthesizer"]
        if intent in {"code_generation", "debugging", "code_review", "code_modification"}:
            if requires_research or evidence_count >= 6:
                return ["researcher", "coder", "synthesizer"]
            return ["coder", "synthesizer"]
        if intent in {"research_summary", "rag_answering"}:
            return ["researcher", "synthesizer"]
        if intent == "document_writing":
            return ["writer", "synthesizer"]
        if intent == "general_qa":
            return ["synthesizer"]
        return ["synthesizer"]

    def should_review(self, session: AgentSession) -> bool:
        return not bool(session.workflow_decision.get("use_quick_answer", False))

    def should_run_quality_gate(self, session: AgentSession) -> bool:
        return self.should_review(session)

    def should_revise(self, session: AgentSession) -> bool:
        gate = session.quality_gate if isinstance(session.quality_gate, dict) else {}
        decision = str(gate.get("decision", "")).lower()
        requires_revision = bool(gate.get("requires_revision", False))
        return decision in {"revise", "fail"} or requires_revision

    def select_revision_target(self, session: AgentSession) -> str | None:
        gate = session.quality_gate if isinstance(session.quality_gate, dict) else {}
        target = gate.get("revision_target_agent")
        if target:
            return str(target)
        if session.review_reports and isinstance(session.review_reports[-1], dict):
            report_target = session.review_reports[-1].get("revision_target_agent")
            if report_target:
                return str(report_target)
        if "synthesizer" in session.selected_agents:
            return "synthesizer"
        return session.selected_agents[0] if session.selected_agents else None

    def max_revision_iterations(self, session: AgentSession) -> int:
        value: Any = session.workflow_decision.get("max_iterations", 1)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

