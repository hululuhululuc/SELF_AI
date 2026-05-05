# coding=utf-8
"""State persistence abstraction for EngineLoop."""

from __future__ import annotations

from typing import Any

from .engine_state import EngineState


class StateStore:
    """In-memory state store with optional ToolRuntime-backed persistence."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._traces: dict[str, list[dict[str, Any]]] = {}
        self._node_outputs: dict[str, dict[str, Any]] = {}
        self._agent_outputs: dict[str, dict[str, Any]] = {}

    async def _runtime_call(
        self,
        run_context: Any | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        if run_context is None:
            return
        tool_runtime = getattr(run_context, "tool_runtime", None)
        if tool_runtime is None:
            return
        try:
            await tool_runtime.execute_by_name(
                tool_name,
                arguments,
                run_id=str(arguments.get("run_id", getattr(run_context, "run_id", ""))),
                session_id=str(getattr(run_context, "session_id", "default")),
                run_context=run_context,
            )
        except Exception:
            return

    async def save_state(self, state: EngineState, *, run_context: Any | None = None) -> None:
        payload = state.to_dict()
        self._states[state.run_id] = payload
        await self._runtime_call(
            run_context,
            "storage.runtime.save_run_state",
            {"run_id": state.run_id, "state": payload},
        )

    async def append_trace(
        self,
        run_id: str,
        trace_item: dict[str, Any],
        *,
        run_context: Any | None = None,
    ) -> None:
        self._traces.setdefault(run_id, []).append(trace_item)
        await self._runtime_call(
            run_context,
            "storage.runtime.append_trace",
            {"run_id": run_id, "trace": trace_item},
        )

    async def save_node_output(
        self,
        run_id: str,
        node_name: str,
        output: dict[str, Any],
        *,
        run_context: Any | None = None,
    ) -> None:
        self._node_outputs[f"{run_id}:{node_name}"] = output
        await self._runtime_call(
            run_context,
            "storage.runtime.save_node_output",
            {"run_id": run_id, "node_name": node_name, "output": output},
        )

    async def save_agent_output(
        self,
        run_id: str,
        agent_name: str,
        output: dict[str, Any],
        *,
        run_context: Any | None = None,
    ) -> None:
        self._agent_outputs[f"{run_id}:{agent_name}"] = output
        await self._runtime_call(
            run_context,
            "storage.runtime.save_agent_output",
            {"run_id": run_id, "agent_name": agent_name, "output": output},
        )

    def get_state(self, run_id: str) -> dict[str, Any] | None:
        return self._states.get(run_id)

    def list_trace(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._traces.get(run_id, []))

    def get_node_output(self, run_id: str, node_name: str) -> dict[str, Any] | None:
        return self._node_outputs.get(f"{run_id}:{node_name}")

    def get_agent_output(self, run_id: str, agent_name: str) -> dict[str, Any] | None:
        return self._agent_outputs.get(f"{run_id}:{agent_name}")
