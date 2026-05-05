# coding=utf-8
"""Tests for memory sidecar agent normalization and decision parsing."""

from __future__ import annotations

import pytest

from self_ai.memory.memory_sidecar import MemorySidecarAgent


def test_memory_sidecar_normalize_decision_clamps_values() -> None:
    agent = MemorySidecarAgent()
    decision = agent.normalize_decision(
        {
            "profiles": {
                "response_summary": {
                    "memory_class": "decision",
                    "importance": 9,
                    "decay_mode": "slow",
                    "pinned": True,
                    "confidence": -1,
                    "decision_source": "model_sidecar_v1",
                }
            }
        }
    )
    profile = decision["profiles"]["response_summary"]
    assert profile["memory_class"] == "decision"
    assert profile["importance"] == 1.0
    assert profile["confidence"] == 0.0
    assert profile["decay_mode"] == "slow"
    assert profile["pinned"] is True


@pytest.mark.asyncio
async def test_memory_sidecar_decide_uses_model_generate_json() -> None:
    agent = MemorySidecarAgent()

    async def _fake_model_generate(_prompt: str) -> dict:
        return {
            "ok": True,
            "data": {
                "response": (
                    '{"profiles":{"turn_summary":{"memory_class":"constraint","importance":0.9,'
                    '"decay_mode":"none","pinned":true,"confidence":0.95}}}'
                )
            },
        }

    decision = await agent.decide(
        task="记住用户偏好",
        response="好的",
        execution_summary={},
        candidates_summary=[],
        recent_turns_summary=[],
        model_generate=_fake_model_generate,
    )
    profile = decision["profiles"]["turn_summary"]
    assert profile["memory_class"] == "constraint"
    assert profile["pinned"] is True
    assert profile["decay_mode"] == "none"


@pytest.mark.asyncio
async def test_memory_sidecar_decide_raises_when_model_call_failed() -> None:
    agent = MemorySidecarAgent()

    async def _fake_model_generate(_prompt: str) -> dict:
        return {"ok": False, "error": {"type": "RateLimitError", "message": "quota exceeded"}}

    with pytest.raises(RuntimeError, match="model call failed"):
        await agent.decide(
            task="remember this",
            response="ok",
            execution_summary={},
            candidates_summary=[],
            recent_turns_summary=[],
            model_generate=_fake_model_generate,
        )
