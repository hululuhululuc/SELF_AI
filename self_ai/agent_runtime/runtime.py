# coding=utf-8
"""Unified AgentRuntime entrypoint."""

from __future__ import annotations

from typing import Any

from .agent_loop import AgentLoop
from .agent_policy import AgentPolicy
from .agent_session import AgentSession
from .message_bus import MessageBus
from .revision_loop import RevisionLoop
from .task_board import TaskBoard


class AgentRuntime:
    """Single entry for multi-agent execution + bounded revision loop."""

    def __init__(
        self,
        *,
        tool_runtime: Any,
        state_store: Any | None = None,
        policy: AgentPolicy | None = None,
        task_board: TaskBoard | None = None,
        message_bus: MessageBus | None = None,
    ) -> None:
        self.tool_runtime = tool_runtime
        self.state_store = state_store
        self.policy = policy or AgentPolicy()
        self.task_board = task_board or TaskBoard()
        self.message_bus = message_bus or MessageBus()
        self.agent_loop = AgentLoop(tool_runtime=tool_runtime, state_store=state_store)
        self.revision_loop = RevisionLoop(
            tool_runtime=tool_runtime,
            agent_loop=self.agent_loop,
            policy=self.policy,
            state_store=state_store,
        )

    async def _trace(
        self,
        session: AgentSession,
        event: str,
        *,
        run_context: Any | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "event": event,
            "run_id": session.run_id,
            "session_id": session.session_id,
            "ts": session.updated_at,
            "fields": fields or {},
        }
        session.metadata.setdefault("trace_events", []).append(payload)
        if self.state_store is not None:
            await self.state_store.append_trace(session.run_id, payload, run_context=run_context)

    async def _save_agent_runtime_output(self, session: AgentSession, *, run_context: Any | None = None) -> None:
        if self.state_store is None:
            return
        await self.state_store.save_node_output(
            session.run_id,
            "agent_run",
            {
                "selected_agents": list(session.selected_agents),
                "agent_count": len(session.selected_agents),
                "output_count": len(session.agent_outputs),
            },
            run_context=run_context,
        )
        await self.state_store.save_node_output(
            session.run_id,
            "agent_runtime",
            {
                "selected_agents": list(session.selected_agents),
                "agent_count": len(session.selected_agents),
                "output_count": len(session.agent_outputs),
                "review_count": len(session.review_reports),
                "quality_gate_decision": session.quality_gate.get("decision")
                if isinstance(session.quality_gate, dict)
                else "",
                "revision_count": session.revision_count,
            },
            run_context=run_context,
        )

    async def run(self, state: Any, *, run_context: Any | None = None) -> AgentSession:
        session = AgentSession.from_engine_state(state)
        session.max_revision_iterations = self.policy.max_revision_iterations(session)
        session.metadata["agent_runtime_used"] = True
        await self._trace(session, "agent_runtime.start", run_context=run_context)

        try:
            if bool(session.workflow_decision.get("use_quick_answer", False)):
                session.selected_agents = []
                session.metadata["agent_runtime_skipped"] = "quick_answer"
                await self._save_agent_runtime_output(session, run_context=run_context)
                await self._trace(
                    session,
                    "agent_runtime.end",
                    run_context=run_context,
                    fields={"skipped": True, "reason": "quick_answer"},
                )
                return session

            selected = self.policy.select_agents(session)
            session.selected_agents = list(selected)
            for agent_name in selected:
                task = self.task_board.create_task(
                    goal=f"Run role agent {agent_name}",
                    agent_name=agent_name,
                    metadata={"run_id": session.run_id},
                )
                session.tasks.append(task)

            await self.agent_loop.run_agents(
                session,
                agents=selected,
                run_context=run_context,
            )

            await self.revision_loop.run_until_complete(
                session,
                run_context=run_context,
                task_board=self.task_board,
                message_bus=self.message_bus,
            )
            await self._save_agent_runtime_output(session, run_context=run_context)
            await self._trace(
                session,
                "agent_runtime.end",
                run_context=run_context,
                fields={
                    "selected_agents": list(session.selected_agents),
                    "output_count": len(session.agent_outputs),
                    "revision_count": session.revision_count,
                },
            )
            return session
        except Exception as exc:
            session.add_error(type(exc).__name__, str(exc), stage="agent_runtime.run")
            await self._trace(
                session,
                "agent_runtime.error",
                run_context=run_context,
                fields={"error": type(exc).__name__},
            )
            return session

