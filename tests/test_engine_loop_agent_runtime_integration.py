# coding=utf-8
"""EngineLoop integration checks for MainLoop semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from self_ai.kernel.engine_state import EngineState
from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_engine_loop_mainloop_completes_and_persists_outputs(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    state = EngineState.from_input("Architecture planning for next release", run_id="r-ar", session_id="s1")
    ctx = kernel.new_run_context(run_id="r-ar", session_id="s1")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.response
    assert isinstance(final.workflow_decision, dict)
    assert isinstance(final.selected_agents, list)
    assert isinstance(final.agent_outputs, dict)
    assert isinstance(final.review_reports, list)
    assert isinstance(final.quality_gate, dict)

    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_finalize" in node_names


@pytest.mark.asyncio
async def test_engine_loop_quick_answer_does_not_require_agent_fields(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)
    state = EngineState.from_input("What is Redis?", run_id="r-q", session_id="s1")
    ctx = kernel.new_run_context(run_id="r-q", session_id="s1")

    final = await kernel.engine_loop.run(state, run_context=ctx)

    assert final.workflow_decision.get("use_quick_answer") is True
    assert final.response
    assert isinstance(final.selected_agents, list)
    assert isinstance(final.agent_outputs, dict)

    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert "mainloop_final_answer" in node_names
    assert "mainloop_finalize" in node_names
