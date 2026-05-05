# coding=utf-8
"""Legacy debate cleanup checks (MainLoop semantics)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests._engine_helpers import build_test_kernel


def test_legacy_debate_module_removed_from_production_path() -> None:
    assert importlib.util.find_spec("self_ai.debate") is None


@pytest.mark.asyncio
async def test_kernel_run_has_no_debate_stage_or_trace(tmp_path: Path) -> None:
    kernel, redis_store = build_test_kernel(tmp_path)

    await kernel.run("architecture tradeoff for distributed system", session_id="s-debate-clean")

    node_names = [name for _, name, _ in redis_store.node_outputs]
    assert all("debate" not in name for name in node_names)

    trace_events = [event.get("event", "") for _, event in redis_store.traces if isinstance(event, dict)]
    assert all("debate" not in event_name for event_name in trace_events)
