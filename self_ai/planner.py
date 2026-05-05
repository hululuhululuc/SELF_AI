# coding=utf-8
"""Planner models and helpers for Phase 5B."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from .execution_modes import ExecutionMode
from .schemas import TaskProfile
from .text_utils import shorten_text

_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)```|```\s*([\s\S]*?)```")


class PlanStep(BaseModel):
    step_id: str
    goal: str
    agent_name: str = "executor"
    depends_on: list[str] = Field(default_factory=list)
    expected_output: str = ""
    done_criteria: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    run_id: str = ""
    session_id: str = "default"
    task_summary: str = ""
    execution_mode: str = ExecutionMode.GENERAL.value
    steps: list[PlanStep] = Field(default_factory=list)
    rationale: str = ""
    planner_version: str = "phase5b.v1"
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


def _normalize_text(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        return str(value.value)
    text = str(value).strip()
    return text if text else default


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _JSON_BLOCK_RE.search(stripped)
    if fenced:
        candidate = fenced.group(1) or fenced.group(2) or ""
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Unable to parse planner JSON response")


def _build_plan_prompt(
    *,
    task_summary: str,
    profile_data: dict[str, Any],
    workflow_decision: dict[str, Any],
) -> str:
    return (
        "You are a planner for a coding/research workflow.\n"
        "Return only valid JSON with keys: task_summary, execution_mode, rationale, steps.\n"
        "Each step must have: step_id, goal, agent_name, depends_on, expected_output, done_criteria.\n"
        "No markdown. No prose outside JSON.\n\n"
        f"Task: {task_summary}\n"
        f"Intent: {_normalize_text(profile_data.get('intent'), default='general_qa')}\n"
        f"Complexity: {_normalize_text(profile_data.get('complexity'), default='medium')}\n"
        f"Risk: {_normalize_text(profile_data.get('risk_level'), default='medium')}\n"
        f"Execution mode: {_normalize_text(workflow_decision.get('execution_mode'), default='general')}\n"
    )


def should_plan(
    *,
    task_profile: TaskProfile | dict[str, Any] | None,
    workflow_decision: dict[str, Any] | None,
) -> bool:
    """Decide whether task needs explicit structured planning."""
    profile = _to_dict(task_profile)
    decision = _to_dict(workflow_decision)

    if bool(decision.get("use_quick_answer", False)):
        return False

    intent = _normalize_text(profile.get("intent"), default="general_qa")
    complexity = _normalize_text(profile.get("complexity"), default="medium")
    risk_level = _normalize_text(profile.get("risk_level"), default="medium")
    execution_mode = _normalize_text(decision.get("execution_mode"), default="general")

    if intent in {"document_writing", "general_qa"} and complexity == "low":
        return False

    if bool(profile.get("requires_planning", False)):
        return True
    if execution_mode == ExecutionMode.ARCHITECTURE_DESIGN.value:
        return True
    if complexity == "high" or risk_level == "high":
        return True
    if intent in {"architecture_design", "planning"}:
        return True
    if intent in {"debugging", "code_review", "code_modification"} and complexity != "low":
        return True
    return False


def fallback_plan(
    *,
    run_id: str = "",
    session_id: str = "default",
    task_summary: str,
    task_profile: TaskProfile | dict[str, Any] | None,
    workflow_decision: dict[str, Any] | None,
    reason: str = "fallback",
) -> Plan:
    """Build deterministic fallback plan when model planning is unavailable."""
    profile = _to_dict(task_profile)
    decision = _to_dict(workflow_decision)
    intent = _normalize_text(profile.get("intent"), default="general_qa")
    execution_mode = _normalize_text(
        decision.get("execution_mode"), default=ExecutionMode.GENERAL.value
    )

    def _step(
        idx: int,
        goal: str,
        *,
        agent: str = "planner",
        depends_on: list[str] | None = None,
        output: str = "",
        done: str = "",
    ) -> PlanStep:
        return PlanStep(
            step_id=f"s{idx}",
            goal=goal,
            agent_name=agent,
            depends_on=depends_on or [],
            expected_output=output,
            done_criteria=done,
        )

    if intent in {"architecture_design", "planning"}:
        steps = [
            _step(1, "Analyze constraints and architecture scope", agent="architect", output="scope + constraints"),
            _step(2, "Design staged implementation plan", agent="architect", depends_on=["s1"], output="phase plan"),
            _step(3, "Define verification milestones and risks", agent="architect", depends_on=["s2"], output="risk checklist"),
        ]
    elif intent in {"debugging", "code_review"}:
        steps = [
            _step(1, "Locate failure points and reproduce issue", agent="debugger", output="failure hypothesis"),
            _step(2, "Inspect code paths and isolate root cause", agent="debugger", depends_on=["s1"], output="root cause"),
            _step(3, "Apply fix and validate with checks/tests", agent="debugger", depends_on=["s2"], output="fix + verification"),
        ]
    elif intent == "code_modification":
        steps = [
            _step(1, "Assess change impact and touched scope", agent="coder", output="impact summary"),
            _step(2, "Implement modifications with minimal side effects", agent="coder", depends_on=["s1"], output="patch"),
            _step(3, "Verify behavior and compatibility", agent="coder", depends_on=["s2"], output="validation notes"),
        ]
    elif intent == "research_summary":
        steps = [
            _step(1, "Collect relevant evidence sources", agent="researcher", output="evidence list"),
            _step(2, "Organize evidence by themes", agent="researcher", depends_on=["s1"], output="theme map"),
            _step(3, "Synthesize concise conclusions", agent="researcher", depends_on=["s2"], output="summary"),
        ]
    elif intent == "document_writing":
        steps = [
            _step(1, "Extract style and constraint requirements", agent="writer", output="style constraints"),
            _step(2, "Draft and polish final text", agent="writer", depends_on=["s1"], output="final document"),
        ]
    else:
        steps = [
            _step(1, "Clarify objective and context", output="task understanding"),
            _step(2, "Produce direct answer with key justification", depends_on=["s1"], output="final answer"),
        ]

    return Plan(
        run_id=run_id,
        session_id=session_id,
        task_summary=shorten_text(task_summary, max_chars=220),
        execution_mode=execution_mode,
        steps=steps,
        rationale=f"Fallback planner used: {reason}",
        metadata={"fallback": True, "intent": intent},
    )


async def build_plan(
    *,
    task_summary: str,
    run_id: str = "",
    session_id: str = "default",
    task_profile: TaskProfile | dict[str, Any] | None,
    workflow_decision: dict[str, Any] | None,
    route_model_func: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
) -> Plan:
    """Build plan from model JSON when available, otherwise fallback safely."""
    profile = _to_dict(task_profile)
    decision = _to_dict(workflow_decision)
    execution_mode = _normalize_text(
        decision.get("execution_mode"), default=ExecutionMode.GENERAL.value
    )
    if route_model_func is None:
        return fallback_plan(
            run_id=run_id,
            session_id=session_id,
            task_summary=task_summary,
            task_profile=task_profile,
            workflow_decision=workflow_decision,
            reason="route_model_func_missing",
        )

    prompt = _build_plan_prompt(
        task_summary=task_summary,
        profile_data=profile,
        workflow_decision=decision,
    )
    try:
        model_result = await route_model_func(prompt, "reasoning")
        model_response = str(model_result.get("response", ""))
        parsed = _extract_json_object(model_response)
        raw_steps = parsed.get("steps", [])
        if not isinstance(raw_steps, list):
            raise ValueError("planner steps must be a list")
        steps: list[PlanStep] = []
        for idx, raw in enumerate(raw_steps, start=1):
            if not isinstance(raw, dict):
                continue
            step = PlanStep(
                step_id=_normalize_text(raw.get("step_id"), default=f"s{idx}"),
                goal=_normalize_text(raw.get("goal"), default=f"step {idx}"),
                agent_name=_normalize_text(raw.get("agent_name"), default="planner"),
                depends_on=list(raw.get("depends_on", []))
                if isinstance(raw.get("depends_on", []), list)
                else [],
                expected_output=_normalize_text(raw.get("expected_output"), default=""),
                done_criteria=_normalize_text(raw.get("done_criteria"), default=""),
                metadata=_to_dict(raw.get("metadata")),
            )
            steps.append(step)
        if not steps:
            raise ValueError("planner produced empty steps")
        return Plan(
            run_id=run_id,
            session_id=session_id,
            task_summary=shorten_text(
                _normalize_text(parsed.get("task_summary"), default=task_summary),
                max_chars=220,
            ),
            execution_mode=_normalize_text(
                parsed.get("execution_mode"), default=execution_mode
            ),
            steps=steps,
            rationale=shorten_text(
                _normalize_text(parsed.get("rationale"), default="model-generated plan"),
                max_chars=500,
            ),
            metadata={"fallback": False},
        )
    except Exception as exc:
        return fallback_plan(
            run_id=run_id,
            session_id=session_id,
            task_summary=task_summary,
            task_profile=task_profile,
            workflow_decision=workflow_decision,
            reason=type(exc).__name__,
        )
