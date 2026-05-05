# coding=utf-8
"""Agent execution loop via ToolRuntime tools only."""

from __future__ import annotations

from typing import Any

from ..text_utils import shorten_text
from .agent_session import AgentSession


class AgentLoop:
    """Runs role agents and synthesizer through ToolRuntime."""

    def __init__(self, *, tool_runtime: Any, state_store: Any | None = None) -> None:
        self.tool_runtime = tool_runtime
        self.state_store = state_store

    def _build_agent_input(
        self,
        session: AgentSession,
        *,
        revision_instruction: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "selected_agents": list(session.selected_agents),
            "revision_count": session.revision_count,
        }
        if revision_instruction:
            metadata["revision_instruction"] = revision_instruction
        if extra_metadata:
            metadata.update(extra_metadata)
        return {
            "run_id": session.run_id,
            "session_id": session.session_id,
            "task": session.task,
            "task_profile": session.task_profile,
            "workflow_decision": session.workflow_decision,
            "plan": session.plan,
            "context_pack": session.context_pack,
            "research_results": (
                list(session.context_pack.get("evidence_items", []))
                if isinstance(session.context_pack.get("evidence_items"), list)
                else []
            ),
            "debate_messages": [],
            "metadata": metadata,
        }

    async def _save_agent_output(
        self,
        session: AgentSession,
        *,
        agent_name: str,
        result: dict[str, Any],
        run_context: Any | None,
    ) -> None:
        if self.state_store is None:
            return
        summary = {
            "agent_name": agent_name,
            "role": result.get("role"),
            "output_type": result.get("output_type"),
            "summary": shorten_text(str(result.get("summary", "")), max_chars=220),
            "confidence": result.get("confidence"),
            "warning_count": len(result.get("warnings", []))
            if isinstance(result.get("warnings", []), list)
            else 0,
            "output_preview": shorten_text(str(result.get("output", "")), max_chars=220),
        }
        await self.state_store.save_agent_output(
            session.run_id,
            agent_name,
            summary,
            run_context=run_context,
        )

    async def run_agent(
        self,
        session: AgentSession,
        *,
        agent_name: str,
        run_context: Any | None = None,
        instruction: str | None = None,
    ) -> bool:
        call_args: dict[str, Any] = {
            "agent_name": agent_name,
            "agent_input": self._build_agent_input(
                session,
                revision_instruction=instruction,
            ),
        }
        if instruction:
            call_args["instruction"] = instruction
        result = await self.tool_runtime.execute_by_name(
            "agent.run",
            call_args,
            run_id=session.run_id,
            session_id=session.session_id,
            run_context=run_context,
            metadata={"component": "agent_runtime", "stage": "agent_loop", "agent_name": agent_name},
        )
        if not result.ok:
            session.add_error(
                result.error.get("type", "AgentRunError") if isinstance(result.error, dict) else "AgentRunError",
                result.error.get("message", "agent run failed") if isinstance(result.error, dict) else "agent run failed",
                stage="agent_loop",
                metadata={"agent_name": agent_name, "tool_name": result.tool_name},
            )
            return False
        payload = result.data.get("agent_result", {})
        if not isinstance(payload, dict):
            session.add_error("AgentRunError", "invalid agent_result payload", stage="agent_loop", metadata={"agent_name": agent_name})
            return False
        session.add_agent_output(agent_name, payload)
        await self._save_agent_output(session, agent_name=agent_name, result=payload, run_context=run_context)
        return True

    async def run_synthesizer(
        self,
        session: AgentSession,
        *,
        run_context: Any | None = None,
        instruction: str | None = None,
    ) -> bool:
        summaries: list[dict[str, Any]] = []
        for name, output in session.agent_outputs.items():
            if not isinstance(output, dict):
                continue
            summaries.append(
                {
                    "agent_name": name,
                    "summary": shorten_text(str(output.get("summary", "")), max_chars=180),
                    "output_preview": shorten_text(str(output.get("output", "")), max_chars=180),
                }
            )
        agent_input = self._build_agent_input(
            session,
            revision_instruction=instruction,
            extra_metadata={"agent_outputs_summary": summaries},
        )
        result = await self.tool_runtime.execute_by_name(
            "agent.synthesize",
            {
                "agent_input": agent_input,
                "agent_outputs_summary": summaries,
                "instruction": instruction or "",
            },
            run_id=session.run_id,
            session_id=session.session_id,
            run_context=run_context,
            metadata={"component": "agent_runtime", "stage": "agent_loop", "agent_name": "synthesizer"},
        )
        if not result.ok:
            session.add_error(
                result.error.get("type", "SynthesizerError") if isinstance(result.error, dict) else "SynthesizerError",
                result.error.get("message", "synthesizer failed") if isinstance(result.error, dict) else "synthesizer failed",
                stage="agent_loop",
            )
            return False
        payload = result.data.get("agent_result", {})
        if not isinstance(payload, dict):
            session.add_error("SynthesizerError", "invalid synthesizer payload", stage="agent_loop")
            return False
        session.add_agent_output("synthesizer", payload)
        await self._save_agent_output(session, agent_name="synthesizer", result=payload, run_context=run_context)
        return True

    async def run_agents(
        self,
        session: AgentSession,
        *,
        agents: list[str],
        run_context: Any | None = None,
    ) -> None:
        non_synth = [a for a in agents if a != "synthesizer"]
        for agent_name in non_synth:
            await self.run_agent(session, agent_name=agent_name, run_context=run_context)
        if "synthesizer" in agents:
            await self.run_synthesizer(session, run_context=run_context)

