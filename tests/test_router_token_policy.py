# coding=utf-8
"""Token policy tests aligned with active model-family output limits."""

from self_ai.router import _default_max_tokens, _model_output_cap, _tier_output_floor


def test_model_output_caps() -> None:
    assert _model_output_cap("deepseek-v4-flash") == 393216
    assert _model_output_cap("deepseek-chat") == 393216
    assert _model_output_cap("deepseek-ai/DeepSeek-V4-Flash") == 393216
    assert _model_output_cap("qwen3.6-plus") == 65536
    assert _model_output_cap("Qwen/Qwen3-32B") == 65536
    assert _model_output_cap("glm-5.1") == 131072
    assert _model_output_cap("kimi-k2.6") == 16384


def test_default_max_tokens_are_not_too_low() -> None:
    assert _default_max_tokens("fast") >= 16384
    assert _default_max_tokens("balanced") >= 32768
    assert _default_max_tokens("high") >= 65536
    assert _default_max_tokens("code") >= 65536


def test_tier_floor_respects_model_caps() -> None:
    assert _tier_output_floor("high", model_name="qwen3.6-plus") == 65536
    assert _tier_output_floor("code", model_name="kimi-k2.6") == 16384
    assert _tier_output_floor("balanced", model_name="glm-5.1") == 32768
    assert _tier_output_floor("balanced", model_name="kimi-k2.6") == 16384
