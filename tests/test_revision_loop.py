# coding=utf-8
"""Tests for RevisionLoop."""

from __future__ import annotations

from typing import Any

import pytest

from self_ai.adapters.agent_tools import register_agent_tools
from self_ai.adapters.model_tool import register_model_tools
from self_ai.adapters.review_tools import register_review_tools
from self_ai.agent_runtime.agent_loop import AgentLoop
from self_ai.agent_runtime.agent_policy import AgentPolicy
from self_ai.agent_runtime.agent_session import AgentSession
from self_ai.kernel.run_context import RunContext
from self_ai.agent_runtime.revision_loop import RevisionLoop
from self_ai.runtime.tool_runtime import ToolRuntime
from self_ai.runtime.tool_result import ToolResult


class _FakeToolRuntime:
    def __init__(self, *, decision: str = "pass") -> None:
        self.decision = decision
        self.calls: list[str] = []

    async def execute_by_name(self, name: str, arguments: dict[str, Any], **_kwargs: Any) -> ToolResult:
        self.calls.append(name)
        if name == "review.run":
            return ToolResult.success(
                call_id="r1",
                tool_name=name,
                data={
                    "review_report": {
                        "pass_review": self.decision in {"pass", "warn"},
                        "severity": "medium",
                        "major_issues": [] if self.decision in {"pass", "warn"} else ["bug"],
                        "minor_issues": ["minor"] if self.decision == "warn" else [],
                        "missing_requirements": [],
                        "requires_revision": self.decision == "revise",
                        "revision_target_agent": "coder",
                        "revision_instruction": "fix bug",
                    }
                },
            )
        if name == "review.quality_gate.decide":
            return ToolResult.success(
                call_id="g1",
                tool_name=name,
                data={
                    "quality_gate": {
                        "decision": self.decision,
                        "passed": self.decision in {"pass", "warn"},
                        "requires_revision": self.decision == "revise",
                        "revision_target_agent": "coder",
                        "revision_instruction": "fix bug",
                    }
                },
            )
        return ToolResult.failure(call_id="x", tool_name=name, error_type="Missing", message="missing")


class _FakeAgentLoop:
    def __init__(self) -> None:
        self.agent_runs: list[str] = []
        self.synth_runs = 0

    async def run_agent(self, session: AgentSession, *, agent_name: str, run_context: Any | None = None, instruction: str | None = None) -> bool:
        self.agent_runs.append(agent_name)
        session.add_agent_output(agent_name, {"agent_name": agent_name, "output": instruction or "updated"})
        return True

    async def run_synthesizer(self, session: AgentSession, *, run_context: Any | None = None, instruction: str | None = None) -> bool:
        self.synth_runs += 1
        session.add_agent_output("synthesizer", {"agent_name": "synthesizer", "output": "merged"})
        return True


@pytest.mark.asyncio
async def test_revision_loop_pass_warn_do_not_iterate() -> None:
    session = AgentSession(
        run_id="r1",
        session_id="s1",
        task="t",
        selected_agents=["coder", "synthesizer"],
        max_revision_iterations=1,
    )
    loop = RevisionLoop(
        tool_runtime=_FakeToolRuntime(decision="warn"),
        agent_loop=_FakeAgentLoop(),
        policy=AgentPolicy(),
    )
    out = await loop.run_until_complete(session)
    assert out.revision_count == 0
    assert out.quality_gate["decision"] == "warn"


@pytest.mark.asyncio
async def test_revision_loop_revise_reruns_target_and_is_bounded() -> None:
    session = AgentSession(
        run_id="r2",
        session_id="s2",
        task="t",
        selected_agents=["coder", "synthesizer"],
        max_revision_iterations=1,
    )
    fake_agent_loop = _FakeAgentLoop()
    loop = RevisionLoop(
        tool_runtime=_FakeToolRuntime(decision="revise"),
        agent_loop=fake_agent_loop,
        policy=AgentPolicy(),
    )
    out = await loop.run_until_complete(session)
    assert out.revision_count == 1
    assert fake_agent_loop.agent_runs == ["coder"]
    assert out.metadata["revision_performed"] is True


@pytest.mark.asyncio
async def test_revision_loop_missing_target_stops_safely() -> None:
    session = AgentSession(
        run_id="r3",
        session_id="s3",
        task="t",
        selected_agents=[],
        max_revision_iterations=1,
    )
    loop = RevisionLoop(
        tool_runtime=_FakeToolRuntime(decision="revise"),
        agent_loop=_FakeAgentLoop(),
        policy=AgentPolicy(),
    )
    out = await loop.run_until_complete(session)
    assert any(err["type"] in {"RevisionTargetMissing", "RevisionTargetInvalid"} for err in out.errors)


@pytest.mark.asyncio
async def test_revision_loop_uses_model_generate_for_review_and_revision_agent(tmp_path) -> None:
    call_categories: list[str] = []

    async def _fake_route_model(prompt: str, category: str) -> dict[str, Any]:
        call_categories.append(category)
        if "storage-driven reviewer" in prompt:
            return {
                "model": "mock-review",
                "response": (
                    '{"pass_review": false, "severity": "high", "major_issues": ["bug"], '
                    '"minor_issues": [], "missing_requirements": [], "requires_revision": true, '
                    '"revision_target_agent": "coder", "revision_instruction": "fix bug", '
                    '"evidence_refs": [], "storage_evidence": [], "rationale": "needs fix"}'
                ),
            }
        return {"model": "mock-agent", "response": "agent output"}

    runtime = ToolRuntime()
    register_model_tools(runtime.registry, route_model_func=_fake_route_model)
    register_agent_tools(runtime.registry)
    register_review_tools(runtime.registry)
    ctx = RunContext(project_root=tmp_path, tool_runtime=runtime, permission_profile={"model_call": True})

    session = AgentSession(
        run_id="r4",
        session_id="s4",
        task="debug this",
        selected_agents=["coder", "synthesizer"],
        max_revision_iterations=1,
        task_profile={"risk_level": "high", "complexity": "high"},
    )

    agent_loop = AgentLoop(tool_runtime=runtime)
    loop = RevisionLoop(tool_runtime=runtime, agent_loop=agent_loop, policy=AgentPolicy())
    out = await loop.run_until_complete(session, run_context=ctx)

    assert out.revision_count == 1
    assert out.quality_gate.get("decision") in {"revise", "fail"}
    # review uses reasoning category, agent reruns use coding/reasoning categories.
    assert "reasoning" in set(call_categories)
    assert "coding" in set(call_categories)
