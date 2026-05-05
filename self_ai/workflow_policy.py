# coding=utf-8
"""Workflow policy decision for Phase 5A conditional routing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .execution_modes import ExecutionMode
from .schemas import RetrievalPolicy, TaskProfile


class WorkflowDecision(BaseModel):
    run_id: str = ""
    session_id: str = "default"
    execution_mode: ExecutionMode = ExecutionMode.GENERAL
    next_node: str = "research"
    skip_research: bool = False
    skip_debate: bool = True
    use_quick_answer: bool = False
    requires_context: bool = True
    requires_debate: bool = False
    max_iterations: int = 1
    selected_agents: list[str] = Field(default_factory=list)
    rationale: str = ""
    policy_version: str = "phase5a.v1"
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


def _normalize_intent(value: Any) -> str:
    if value is None:
        return "general_qa"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _normalize_level(value: Any, *, default: str = "medium") -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        return str(value.value)
    text = str(value).strip().lower()
    return text if text else default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _profile_and_policy(
    *,
    state: dict[str, Any] | None,
    task_profile: TaskProfile | dict[str, Any] | None,
    retrieval_policy: RetrievalPolicy | dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_profile = _to_dict((state or {}).get("task_profile")) if state else {}
    profile_data = _to_dict(task_profile) or state_profile

    state_policy = _to_dict((state or {}).get("retrieval_policy")) if state else {}
    profile_policy = _to_dict(profile_data.get("retrieval_policy"))
    policy_data = _to_dict(retrieval_policy) or state_policy or profile_policy
    return profile_data, policy_data


def _decision_summary(
    *,
    run_id: str,
    session_id: str,
    execution_mode: ExecutionMode,
    next_node: str,
    skip_research: bool,
    skip_debate: bool,
    use_quick_answer: bool,
    requires_context: bool,
    requires_debate: bool,
    max_iterations: int,
    selected_agents: list[str],
    rationale: str,
    metadata: dict[str, Any] | None = None,
) -> WorkflowDecision:
    return WorkflowDecision(
        run_id=run_id,
        session_id=session_id,
        execution_mode=execution_mode,
        next_node=next_node,
        skip_research=skip_research,
        skip_debate=skip_debate,
        use_quick_answer=use_quick_answer,
        requires_context=requires_context,
        requires_debate=requires_debate,
        max_iterations=max(1, int(max_iterations)),
        selected_agents=selected_agents,
        rationale=rationale[:500],
        metadata=metadata or {},
    )


def decide_workflow(
    *,
    state: dict[str, Any] | None = None,
    task_profile: TaskProfile | dict[str, Any] | None = None,
    retrieval_policy: RetrievalPolicy | dict[str, Any] | None = None,
) -> WorkflowDecision:
    """Decide minimal workflow routing based on TaskProfile + RetrievalPolicy."""
    try:
        profile_data, policy_data = _profile_and_policy(
            state=state,
            task_profile=task_profile,
            retrieval_policy=retrieval_policy,
        )

        identity = _to_dict(profile_data.get("identity"))
        run_id = str(
            (state or {}).get("run_id")
            or identity.get("run_id")
            or ""
        )
        session_id = str(
            (state or {}).get("session_id")
            or identity.get("session_id")
            or "default"
        )

        intent = _normalize_intent(profile_data.get("intent"))
        complexity = _normalize_level(profile_data.get("complexity"), default="medium")
        risk_level = _normalize_level(profile_data.get("risk_level"), default="medium")
        requires_code = _bool(profile_data.get("requires_code"), False)
        requires_research = _bool(profile_data.get("requires_research"), False)
        requires_planning = _bool(profile_data.get("requires_planning"), False)
        requires_review = _bool(profile_data.get("requires_review"), False)
        use_graph = _bool(policy_data.get("use_graph"), True)

        if (
            intent == "general_qa"
            and complexity == "low"
            and not requires_research
            and not requires_code
        ):
            return _decision_summary(
                run_id=run_id,
                session_id=session_id,
                execution_mode=ExecutionMode.QUICK_ANSWER,
                next_node="quick_answer",
                skip_research=True,
                skip_debate=True,
                use_quick_answer=True,
                requires_context=False,
                requires_debate=False,
                max_iterations=1,
                selected_agents=["responder"],
                rationale="general_qa with low complexity and no research/code requirement",
                metadata={"intent": intent, "complexity": complexity, "risk_level": risk_level},
            )

        if intent == "document_writing":
            skip_research = not requires_research
            return _decision_summary(
                run_id=run_id,
                session_id=session_id,
                execution_mode=ExecutionMode.DOCUMENT_WRITING,
                next_node="implement" if skip_research else "research",
                skip_research=skip_research,
                skip_debate=True,
                use_quick_answer=False,
                requires_context=not skip_research,
                requires_debate=False,
                max_iterations=1,
                selected_agents=["writer"],
                rationale=(
                    "document_writing path; skip debate and only research when explicitly required"
                ),
                metadata={"intent": intent, "requires_research": requires_research},
            )

        if intent in {"research_summary", "rag_answering"}:
            return _decision_summary(
                run_id=run_id,
                session_id=session_id,
                execution_mode=ExecutionMode.RESEARCH_THEN_IMPLEMENT,
                next_node="research",
                skip_research=False,
                skip_debate=True,
                use_quick_answer=False,
                requires_context=True,
                requires_debate=False,
                max_iterations=1,
                selected_agents=["researcher", "implementer"],
                rationale="research intent requires retrieval and skips debate",
                metadata={"intent": intent, "use_graph": use_graph},
            )

        if intent == "code_generation":
            is_low = complexity == "low" and risk_level == "low"
            return _decision_summary(
                run_id=run_id,
                session_id=session_id,
                execution_mode=ExecutionMode.CODE_FOCUSED,
                next_node="research",
                skip_research=False,
                skip_debate=is_low,
                use_quick_answer=False,
                requires_context=True,
                requires_debate=not is_low,
                max_iterations=1,
                selected_agents=["coder"] if is_low else ["coder", "debater"],
                rationale="code_generation path with debate gated by low complexity/risk",
                metadata={
                    "intent": intent,
                    "complexity": complexity,
                    "risk_level": risk_level,
                },
            )

        if intent in {"debugging", "code_review", "code_modification"}:
            needs_debate = (complexity in {"medium", "high"}) or (risk_level != "low")
            return _decision_summary(
                run_id=run_id,
                session_id=session_id,
                execution_mode=ExecutionMode.CODE_FOCUSED,
                next_node="research",
                skip_research=False,
                skip_debate=not needs_debate,
                use_quick_answer=False,
                requires_context=True,
                requires_debate=needs_debate,
                max_iterations=2 if needs_debate else 1,
                selected_agents=["debugger"] if not needs_debate else ["debugger", "debater"],
                rationale="code diagnosis/review path with debate for medium/high complexity or elevated risk",
                metadata={
                    "intent": intent,
                    "complexity": complexity,
                    "risk_level": risk_level,
                    "requires_review": requires_review,
                },
            )

        if intent in {"architecture_design", "planning"} or requires_planning:
            return _decision_summary(
                run_id=run_id,
                session_id=session_id,
                execution_mode=ExecutionMode.ARCHITECTURE_DESIGN,
                next_node="research",
                skip_research=False,
                skip_debate=False,
                use_quick_answer=False,
                requires_context=True,
                requires_debate=True,
                max_iterations=2,
                selected_agents=["architect", "researcher", "debater", "implementer"],
                rationale="architecture/planning path requires context and debate",
                metadata={
                    "intent": intent,
                    "complexity": complexity,
                    "risk_level": risk_level,
                },
            )

        return _decision_summary(
            run_id=run_id,
            session_id=session_id,
            execution_mode=ExecutionMode.GENERAL,
            next_node="research",
            skip_research=False,
            skip_debate=True,
            use_quick_answer=False,
            requires_context=True,
            requires_debate=False,
            max_iterations=1,
            selected_agents=["implementer"],
            rationale="safe default: research then implement with debate skipped",
            metadata={
                "intent": intent,
                "complexity": complexity,
                "risk_level": risk_level,
            },
        )
    except Exception:
        # Safety-first fallback: never interrupt workflow due to policy decision errors.
        state_data = state or {}
        return _decision_summary(
            run_id=str(state_data.get("run_id", "")),
            session_id=str(state_data.get("session_id", "default")),
            execution_mode=ExecutionMode.GENERAL,
            next_node="research",
            skip_research=False,
            skip_debate=True,
            use_quick_answer=False,
            requires_context=True,
            requires_debate=False,
            max_iterations=1,
            selected_agents=["implementer"],
            rationale="policy fallback on unexpected error",
            metadata={"fallback": True},
        )
