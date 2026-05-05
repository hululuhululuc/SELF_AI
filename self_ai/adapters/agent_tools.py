# coding=utf-8
"""Role-agent tool adapters using model.generate as the only model path."""

from __future__ import annotations

from typing import Any

from ..agents import AgentInput, ArchitectAgent, CoderAgent, ResearcherAgent, SynthesizerAgent, WriterAgent
from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec


def _default_agent_map() -> dict[str, Any]:
    return {
        "coder": CoderAgent(),
        "architect": ArchitectAgent(),
        "researcher": ResearcherAgent(),
        "writer": WriterAgent(),
        "synthesizer": SynthesizerAgent(),
    }


def _category_for_agent(agent_name: str) -> str:
    mapping = {
        "coder": "coding",
        "architect": "reasoning",
        "researcher": "research",
        "writer": "fast",
        "synthesizer": "reasoning",
    }
    return mapping.get(agent_name, "fast")


def _require_tool_runtime(run_context: Any | None) -> Any:
    if run_context is None:
        raise RuntimeError("agent tools require run_context with tool_runtime")
    runtime = getattr(run_context, "tool_runtime", None)
    if runtime is None:
        raise RuntimeError("agent tools require run_context.tool_runtime")
    return runtime


def _coerce_agent_input(raw_input: Any, *, fallback: dict[str, Any] | None = None) -> AgentInput:
    payload = dict(fallback or {})
    if isinstance(raw_input, AgentInput):
        return raw_input
    if isinstance(raw_input, dict):
        payload.update(raw_input)
    return AgentInput(**payload)


def register_agent_tools(
    registry: ToolRegistry,
    *,
    agent_map: dict[str, Any] | None = None,
    route_model_func: Any | None = None,
) -> None:
    """Register role-agent wrappers.

    `route_model_func` is kept only for backwards compatibility with existing
    dependency injection signatures. It is intentionally unused here because
    agent/review model calls must go through the model.generate tool path.
    """

    _ = route_model_func
    available_agents = dict(agent_map or _default_agent_map())

    async def _agent_run(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        runtime = _require_tool_runtime(run_context)
        name = str(args.get("agent_name", "")).strip().lower()
        if not name:
            raise ValueError("agent.run requires agent_name")
        agent = available_agents.get(name)
        if agent is None:
            raise ValueError(f"unknown agent: {name}")

        raw_input = args.get("agent_input", {})
        instruction = str(args.get("instruction", "") or "").strip()
        agent_input = _coerce_agent_input(raw_input)
        prompt = agent.build_prompt(agent_input)
        if instruction:
            prompt = (
                prompt
                + "\n\nRevision instruction:\n"
                + instruction
                + "\nFollow this revision instruction strictly."
            )

        model_result = await runtime.execute_by_name(
            "model.generate",
            {
                "prompt": prompt,
                "category": _category_for_agent(name),
            },
            run_id=agent_input.run_id or getattr(run_context, "run_id", ""),
            session_id=agent_input.session_id or getattr(run_context, "session_id", "default"),
            run_context=run_context,
            metadata={"component": "agent_tools", "agent_name": name},
        )
        if not model_result.ok:
            error = model_result.error or {}
            raise RuntimeError(f"agent model call failed: {error.get('type', 'ModelError')}")

        data = model_result.data if isinstance(model_result.data, dict) else {}
        output = str(data.get("response", "") or "")
        agent_result = {
            "run_id": agent_input.run_id,
            "session_id": agent_input.session_id,
            "agent_name": name,
            "role": getattr(agent, "role", name),
            "output": output,
            "output_type": "text",
            "summary": output[:180],
            "confidence": 0.75 if output else 0.3,
            "used_context": [
                key
                for key, enabled in {
                    "context_pack": bool(agent_input.context_pack),
                    "plan": bool(agent_input.plan),
                    "research_results": bool(agent_input.research_results),
                }.items()
                if enabled
            ],
            "warnings": [],
            "metadata": {
                "model": str(data.get("model", "")),
                "model_category": _category_for_agent(name),
                "instruction_used": bool(instruction),
            },
        }
        return {
            "agent_result": agent_result,
            "success": True,
            "ok": True,
            "model": agent_result["metadata"]["model"],
            "error": None,
        }

    async def _agent_synthesize(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        runtime = _require_tool_runtime(run_context)
        synth = available_agents.get("synthesizer")
        if synth is None:
            raise ValueError("synthesizer agent not configured")

        raw_input = args.get("agent_input", {})
        fallback = {"metadata": {"synthesis_mode": True}}
        agent_input = _coerce_agent_input(raw_input, fallback=fallback)
        summary = args.get("agent_outputs_summary", [])
        instruction = str(args.get("instruction", "") or "").strip()

        prompt = synth.build_prompt(agent_input)
        if isinstance(summary, list) and summary:
            prompt += "\n\nAgent outputs summary:\n" + "\n".join(
                f"- {str(item)[:200]}" for item in summary[:20]
            )
        if instruction:
            prompt += "\n\nRevision instruction:\n" + instruction

        model_result = await runtime.execute_by_name(
            "model.generate",
            {
                "prompt": prompt,
                "category": _category_for_agent("synthesizer"),
            },
            run_id=agent_input.run_id or getattr(run_context, "run_id", ""),
            session_id=agent_input.session_id or getattr(run_context, "session_id", "default"),
            run_context=run_context,
            metadata={"component": "agent_tools", "agent_name": "synthesizer"},
        )
        if not model_result.ok:
            error = model_result.error or {}
            raise RuntimeError(f"synthesizer model call failed: {error.get('type', 'ModelError')}")

        data = model_result.data if isinstance(model_result.data, dict) else {}
        output = str(data.get("response", "") or "")
        agent_result = {
            "run_id": agent_input.run_id,
            "session_id": agent_input.session_id,
            "agent_name": "synthesizer",
            "role": getattr(synth, "role", "synthesizer"),
            "output": output,
            "output_type": "text",
            "summary": output[:180],
            "confidence": 0.8 if output else 0.35,
            "used_context": ["agent_outputs_summary", "context_pack", "plan"],
            "warnings": [],
            "metadata": {
                "model": str(data.get("model", "")),
                "model_category": _category_for_agent("synthesizer"),
                "instruction_used": bool(instruction),
            },
        }
        return {
            "agent_result": agent_result,
            "success": True,
            "ok": True,
            "model": agent_result["metadata"]["model"],
            "error": None,
        }

    registry.register(
        ToolSpec(
            name="agent.run",
            description="Run one configured role agent via model.generate and return AgentResult.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string"},
                    "agent_input": {"type": "object"},
                    "instruction": {"type": "string"},
                },
                "required": ["agent_name", "agent_input"],
            },
            output_schema={"type": "object", "properties": {"agent_result": {"type": "object"}}},
            permission="model_call",
            timeout_s=120,
            tags=["agent", "role"],
            handler=_agent_run,
        )
    )
    registry.register(
        ToolSpec(
            name="agent.synthesize",
            description="Run synthesizer role agent via model.generate and return synthesis AgentResult.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_input": {"type": "object"},
                    "agent_outputs_summary": {"type": "array"},
                    "instruction": {"type": "string"},
                },
                "required": ["agent_input"],
            },
            output_schema={"type": "object", "properties": {"agent_result": {"type": "object"}}},
            permission="model_call",
            timeout_s=120,
            tags=["agent", "synthesizer"],
            handler=_agent_synthesize,
        )
    )

