# coding=utf-8
"""Reviewer and quality-gate tool adapters using model.generate path."""

from __future__ import annotations

from typing import Any

from ..agents import ReviewerAgent, ReviewerInput
from ..agents.reviewer import _extract_json
from ..quality_gate import decide_quality_gate
from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec


def _require_tool_runtime(run_context: Any | None) -> Any:
    if run_context is None:
        raise RuntimeError("review tools require run_context with tool_runtime")
    runtime = getattr(run_context, "tool_runtime", None)
    if runtime is None:
        raise RuntimeError("review tools require run_context.tool_runtime")
    return runtime


def register_review_tools(
    registry: ToolRegistry,
    *,
    reviewer: ReviewerAgent | None = None,
    route_model_func: Any | None = None,
) -> None:
    """Register review.run and review.quality_gate.decide tools.

    `route_model_func` remains for signature compatibility, but model calls are
    routed via model.generate tool only.
    """

    _ = route_model_func
    reviewer_agent = reviewer or ReviewerAgent()

    async def _review_run(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        runtime = _require_tool_runtime(run_context)

        raw = args.get("review_input", {})
        if (not isinstance(raw, dict) or not raw) and isinstance(args.get("session"), dict):
            session = args.get("session", {})
            raw = {
                "run_id": session.get("run_id", ""),
                "session_id": session.get("session_id", "default"),
                "task": session.get("task", ""),
                "task_profile": session.get("task_profile", {}),
                "workflow_decision": session.get("workflow_decision", {}),
                "plan": session.get("plan", {}),
                "context_pack": session.get("context_pack", {}),
                "agent_outputs": session.get("agent_outputs", {}),
                "redis_context": {},
                "qdrant_review_memory": [],
                "neo4j_graph_context": [],
                "metadata": session.get("metadata", {}) if isinstance(session.get("metadata"), dict) else {},
            }

        review_input = raw if isinstance(raw, ReviewerInput) else ReviewerInput(**raw)
        prompt = reviewer_agent.build_prompt(review_input)
        model_result = await runtime.execute_by_name(
            "model.generate",
            {"prompt": prompt, "category": "reasoning"},
            run_id=review_input.run_id or getattr(run_context, "run_id", ""),
            session_id=review_input.session_id or getattr(run_context, "session_id", "default"),
            run_context=run_context,
            metadata={"component": "review_tools", "stage": "review"},
        )

        if not model_result.ok:
            error = model_result.error or {}
            report = reviewer_agent._fallback_report(  # noqa: SLF001 - controlled adapter reuse
                review_input,
                error_type=str(error.get("type", "ModelGenerateError")),
            )
            return {"review_report": report}

        data = model_result.data if isinstance(model_result.data, dict) else {}
        try:
            raw_report = _extract_json(str(data.get("response", "")))
            parsed = reviewer_agent._normalize_report(review_input, raw_report)  # noqa: SLF001
        except Exception as exc:
            parsed = reviewer_agent._fallback_report(  # noqa: SLF001
                review_input,
                error_type=type(exc).__name__,
            )

        if isinstance(parsed, dict):
            parsed.setdefault("metadata", {})
            if isinstance(parsed["metadata"], dict):
                parsed["metadata"]["model"] = str(data.get("model", ""))
        return {"review_report": parsed if isinstance(parsed, dict) else {}}

    def _quality_gate_decide(args: dict[str, Any], _run_context: Any | None) -> dict[str, Any]:
        decision = decide_quality_gate(
            review_report=args.get("review_report", {}),
            task_profile=args.get("task_profile", {}),
            run_id=str(args.get("run_id", "")),
            session_id=str(args.get("session_id", "default")),
            max_revision_iterations=int(args.get("max_revision_iterations", 1)),
            current_iteration=int(args.get("current_iteration", 0)),
        )
        data = decision.model_dump(mode="json") if hasattr(decision, "model_dump") else decision
        return {"quality_gate": data}

    registry.register(
        ToolSpec(
            name="review.run",
            description="Run storage-driven reviewer via model.generate and return ReviewReport dict.",
            input_schema={
                "type": "object",
                "properties": {
                    "review_input": {"type": "object"},
                    "session": {"type": "object"},
                },
                "required": [],
            },
            output_schema={"type": "object", "properties": {"review_report": {"type": "object"}}},
            permission="model_call",
            timeout_s=90,
            tags=["reviewer"],
            handler=_review_run,
        )
    )
    registry.register(
        ToolSpec(
            name="review.quality_gate.decide",
            description="Decide pass/warn/revise/fail from latest review report.",
            input_schema={
                "type": "object",
                "properties": {
                    "review_report": {"type": "object"},
                    "task_profile": {"type": "object"},
                    "run_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "max_revision_iterations": {"type": "integer"},
                    "current_iteration": {"type": "integer"},
                },
                "required": ["review_report", "task_profile"],
            },
            output_schema={"type": "object", "properties": {"quality_gate": {"type": "object"}}},
            permission="read_only",
            timeout_s=10,
            tags=["quality_gate"],
            handler=_quality_gate_decide,
        )
    )
