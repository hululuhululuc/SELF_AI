# coding=utf-8
"""Model gateway and model-led routing for Self AI.

This module intentionally avoids hard-coded task->model routing tables.
The router asks a selector model to decide tier/latency budget and then
executes the chosen model via DashScope OpenAI-compatible API.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI

from .config import settings
from .observability import trace

_dashscope: AsyncOpenAI | None = None


def _get_dashscope_client() -> AsyncOpenAI:
    """Create the DashScope client only when a real model call is requested."""
    global _dashscope
    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for model.generate calls.")
    if _dashscope is None:
        _dashscope = AsyncOpenAI(
            base_url=settings.dashscope_base_url,
            api_key=settings.dashscope_api_key,
        )
    return _dashscope

TIER_TO_MODEL: dict[str, str] = {
    "fast": settings.model_fast,
    "balanced": settings.model_balanced,
    "high": settings.model_high,
    "code": settings.model_code,
}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(value)


def _extract_content(response: Any) -> tuple[str, str, bool]:
    choices = getattr(response, "choices", None)
    if not choices:
        return "", "", True
    message = getattr(choices[0], "message", None)
    if message is None:
        return "", "", True
    content = _coerce_text(getattr(message, "content", ""))
    reasoning = _coerce_text(getattr(message, "reasoning_content", ""))
    finish_reason = str(getattr(choices[0], "finish_reason", "") or "")
    final = content or reasoning
    return final, finish_reason, False


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _clamp_int(value: Any, *, lower: int, upper: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(lower, min(upper, number))


def _normalize_tier(value: Any) -> str:
    tier = str(value or "").strip().lower()
    return tier if tier in TIER_TO_MODEL else "balanced"


def _default_max_tokens(tier: str) -> int:
    if tier == "fast":
        return 16384
    if tier == "balanced":
        return 32768
    if tier in {"high", "code"}:
        return 65536
    return 32768


def _model_output_cap(model_name: str) -> int:
    name = str(model_name or "").strip().lower()
    # DeepSeek family: output ceiling 384K.
    if "deepseek" in name:
        return 393216
    # Qwen family: output ceiling 64K
    if "qwen" in name:
        return 65536
    # GLM family: output ceiling 128K
    if "glm" in name:
        return 131072
    # Kimi K2.6 family: output ceiling 16K
    if "kimi" in name:
        return 16384
    return 65536


def _tier_output_floor(tier: str, *, model_name: str) -> int:
    cap = _model_output_cap(model_name)
    if tier == "fast":
        return min(16384, cap)
    if tier == "balanced":
        return min(32768, cap)
    if tier == "code":
        return min(65536, cap)
    if tier == "high":
        return min(65536, cap)
    return min(32768, cap)


def _build_selector_prompt(task: str, stage_hint: str) -> str:
    return (
        "You are the model router for an agentic coding system.\n"
        "Choose the minimal tier needed for the task quality.\n"
        "Return JSON only with keys: tier, enable_thinking, max_tokens, reason.\n"
        "tier must be one of: fast, balanced, high, code.\n"
        "Rules:\n"
        "- prefer fast if the task is simple and low risk\n"
        "- use balanced for normal multi-step tasks\n"
        "- use high for hard reasoning / architecture / ambiguous requests\n"
        "- use code for code-heavy generation/refactor/debug tasks\n"
        "- keep max_tokens practical; avoid over-allocation\n"
        f"stage_hint: {stage_hint}\n"
        f"task:\n{task}\n"
    )


def _build_generation_messages(task: str, stage_hint: str) -> list[dict[str, str]]:
    system = (
        "You are Self AI execution model.\n"
        "Be accurate and concise. Follow user intent strictly.\n"
        "Avoid unnecessary verbosity and avoid fabricating facts.\n"
        f"Current stage hint: {stage_hint}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": task}]


async def _select_route_plan(task: str, stage_hint: str) -> dict[str, Any]:
    prompt = _build_selector_prompt(task, stage_hint)
    started = time.perf_counter()
    selector_model = settings.router_selector_model or settings.model_fast
    response = await _get_dashscope_client().chat.completions.create(
        model=selector_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        timeout=min(20, settings.dashscope_timeout_s),
        extra_body={"enable_thinking": False},
    )
    content, _, malformed = _extract_content(response)
    if malformed:
        raise RuntimeError("selector malformed response (choices/message missing)")
    if not content:
        raise RuntimeError("selector empty content")
    parsed = _extract_json_object(content)
    if not parsed:
        raise RuntimeError("selector non-json response")

    tier = _normalize_tier(parsed.get("tier"))
    enable_thinking = bool(
        parsed.get("enable_thinking", settings.dashscope_enable_thinking)
    )
    max_tokens = _clamp_int(
        parsed.get("max_tokens"),
        lower=256,
        upper=393216,
        default=_default_max_tokens(tier),
    )
    reason = str(parsed.get("reason", ""))[:240]
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    trace(
        "route_model.selector.end",
        selector_model=selector_model,
        tier=tier,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        elapsed_ms=elapsed_ms,
        reason_preview=reason[:120],
    )
    return {
        "tier": tier,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "reason": reason,
        "selector_model": selector_model,
    }


async def route_model(
    task: str,
    category: str = "general",
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route and execute one model call.

    Args:
        task: user/system prompt to execute.
        category: treated as stage hint only, not a hard model router.
    """
    stage_hint = str(category or "general")
    route_hints = hints if isinstance(hints, dict) else {}
    client = _get_dashscope_client()
    trace("route_model.start", stage_hint=stage_hint, task_preview=task[:120])

    plan: dict[str, Any]
    forced_tier_raw = route_hints.get("force_tier") if isinstance(route_hints, dict) else None
    forced_tier = _normalize_tier(forced_tier_raw) if forced_tier_raw else ""
    if settings.router_use_selector:
        try:
            selector_task = task
            if route_hints:
                selector_task = (
                    task
                    + "\n\n[router_hints]\n"
                    + json.dumps(route_hints, ensure_ascii=False, default=str)
                )
            plan = await _select_route_plan(selector_task, stage_hint)
        except Exception as exc:
            trace(
                "route_model.selector.failed",
                stage_hint=stage_hint,
                error=type(exc).__name__,
                message=str(exc)[:200],
            )
            plan = {
                "tier": settings.router_default_tier,
                "enable_thinking": settings.router_default_enable_thinking,
                "max_tokens": _default_max_tokens(settings.router_default_tier),
                "reason": "selector_failed_default_plan",
                "selector_model": settings.router_selector_model or settings.model_fast,
            }
    else:
        plan = {
            "tier": settings.router_default_tier,
            "enable_thinking": settings.router_default_enable_thinking,
            "max_tokens": _default_max_tokens(settings.router_default_tier),
            "reason": "selector_disabled_default_plan",
            "selector_model": settings.router_selector_model or settings.model_fast,
        }

    tier = _normalize_tier(plan.get("tier"))
    if forced_tier in TIER_TO_MODEL:
        trace(
            "route_model.tier_forced",
            stage_hint=stage_hint,
            from_tier=tier,
            to_tier=forced_tier,
        )
        tier = forced_tier
    model = TIER_TO_MODEL.get(tier, settings.model_balanced)
    model_cap = _model_output_cap(model)
    tier_floor = _tier_output_floor(tier, model_name=model)
    max_tokens = _clamp_int(
        plan.get("max_tokens"),
        lower=tier_floor,
        upper=model_cap,
        default=min(_default_max_tokens(tier), model_cap),
    )
    hint_max_tokens = route_hints.get("max_tokens") if isinstance(route_hints, dict) else None
    if hint_max_tokens is not None:
        max_tokens = _clamp_int(
            hint_max_tokens,
            lower=256,
            upper=model_cap,
            default=max_tokens,
        )
    enable_thinking = bool(plan.get("enable_thinking", settings.dashscope_enable_thinking))
    if isinstance(route_hints, dict) and "enable_thinking" in route_hints:
        enable_thinking = bool(route_hints.get("enable_thinking"))
    if route_hints:
        task = (
            task
            + "\n\n[execution_hints]\n"
            + json.dumps(route_hints, ensure_ascii=False, default=str)
        )
    messages = _build_generation_messages(task, stage_hint)

    last_error: Exception | None = None
    retry_count = max(1, int(settings.router_max_retries))
    for attempt in range(1, retry_count + 1):
        started = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=settings.dashscope_timeout_s,
                extra_body={"enable_thinking": enable_thinking},
            )
            content, finish_reason, malformed = _extract_content(response)
            if malformed:
                raise RuntimeError(f"Malformed model response: {model}")
            if not content:
                raise RuntimeError(f"Empty model response: {model}")

            trace(
                "route_model.end",
                stage_hint=stage_hint,
                model=model,
                tier=tier,
                attempt=attempt,
                max_tokens=max_tokens,
                timeout_s=settings.dashscope_timeout_s,
                enable_thinking=enable_thinking,
                finish_reason=finish_reason,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                response_preview=content[:160],
            )
            return {"model": model, "response": content}
        except Exception as exc:
            last_error = exc
            trace(
                "route_model.candidate_failed",
                stage_hint=stage_hint,
                model=model,
                tier=tier,
                attempt=attempt,
                error=type(exc).__name__,
                message=str(exc)[:200],
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

    if last_error:
        raise last_error
    raise RuntimeError("route_model failed without explicit error")
