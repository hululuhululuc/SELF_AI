# coding=utf-8
"""ToolResult merger into EngineState."""

from __future__ import annotations

from typing import Any

from ..runtime.tool_result import ToolResult
from ..text_utils import shorten_text
from .engine_state import EngineState


class ResultAggregator:
    """Merge tool results into EngineState with safe failure handling."""

    @staticmethod
    def _tool_error_summary(result: ToolResult) -> dict[str, Any]:
        err = result.error or {}
        return {
            "tool": result.tool_name,
            "type": err.get("type", "ToolError"),
            "message": str(err.get("message", "tool failed"))[:300],
            "latency_ms": result.latency_ms,
            "metadata": result.metadata,
        }

    def merge_tool_result(self, state: EngineState, result: ToolResult, stage: str) -> EngineState:
        if not result.ok:
            summary = self._tool_error_summary(result)
            state.add_error(summary["type"], summary["message"], stage=stage, metadata={"tool": result.tool_name, **result.metadata})
            return state

        tool_name = result.tool_name
        data = result.data or {}

        if tool_name == "model.generate":
            outputs = state.metadata.setdefault("model_outputs", {})
            if isinstance(outputs, dict):
                outputs[stage] = {
                    "model": str(data.get("model", "")),
                    "response_preview": shorten_text(str(data.get("response", "")), max_chars=300),
                }
            if stage in {"quick_answer", "final_response"}:
                state.model = str(data.get("model", ""))
                state.response = str(data.get("response", ""))

        elif tool_name == "storage.semantic.search":
            evidence = data.get("evidence", [])
            if isinstance(evidence, list):
                for item in evidence:
                    if isinstance(item, dict):
                        state.research_results.append(item)

        elif tool_name.startswith("storage.graph."):
            graph_items = state.metadata.setdefault("graph_context", [])
            if isinstance(graph_items, list):
                payload = data.get("paths", [])
                if isinstance(payload, list):
                    graph_items.extend([p for p in payload if isinstance(p, dict)])

        elif tool_name in {"agent.run", "agent.synthesize"}:
            raw_agent = data.get("agent_result", {})
            if isinstance(raw_agent, dict):
                name = str(raw_agent.get("agent_name", "")).strip()
                if name:
                    state.agent_outputs[name] = raw_agent

        elif tool_name == "review.run":
            report = data.get("review_report", {})
            if isinstance(report, dict):
                state.review_reports.append(report)

        elif tool_name == "review.quality_gate.decide":
            gate = data.get("quality_gate", {})
            if isinstance(gate, dict):
                state.quality_gate = gate

        return state

    def merge_agent_session(self, state: EngineState, session: Any) -> EngineState:
        updates: dict[str, Any]
        if isinstance(session, dict):
            updates = session
        elif hasattr(session, "to_engine_state_updates"):
            updates = session.to_engine_state_updates()
        elif hasattr(session, "model_dump"):
            updates = session.model_dump(mode="json")
        else:
            updates = {}

        selected_agents = updates.get("selected_agents")
        if isinstance(selected_agents, list):
            state.selected_agents = [str(agent) for agent in selected_agents]

        agent_outputs = updates.get("agent_outputs")
        if isinstance(agent_outputs, dict):
            state.agent_outputs = agent_outputs

        review_reports = updates.get("review_reports")
        if isinstance(review_reports, list):
            state.review_reports = [r for r in review_reports if isinstance(r, dict)]

        quality_gate = updates.get("quality_gate")
        if isinstance(quality_gate, dict):
            state.quality_gate = quality_gate

        errors = updates.get("errors")
        if isinstance(errors, list):
            for item in errors:
                if not isinstance(item, dict):
                    continue
                state.add_error(
                    str(item.get("type", "AgentRuntimeError")),
                    str(item.get("message", "agent runtime error")),
                    stage=item.get("stage"),
                    metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
                )

        metadata = updates.get("metadata")
        if isinstance(metadata, dict):
            state.metadata.update(metadata)

        updated_at = updates.get("updated_at")
        if isinstance(updated_at, str) and updated_at:
            state.updated_at = updated_at

        return state
