# coding=utf-8
"""Tool executor with timeout, event emission, and safe result wrapping."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..kernel.engine_events import EngineEvent
from .tool_call import ToolCall
from .tool_result import ToolResult
from .tool_spec import ToolSpec

EventCallback = Callable[[EngineEvent], None | Awaitable[None]]


class ToolExecutor:
    """Executes ToolSpec handlers and converts outcomes to ToolResult."""

    def __init__(self, *, event_callback: EventCallback | None = None) -> None:
        self.event_callback = event_callback

    async def _emit(self, event: EngineEvent) -> None:
        if self.event_callback is None:
            return
        try:
            maybe = self.event_callback(event)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception:
            return

    async def _invoke(self, spec: ToolSpec, arguments: dict[str, Any], run_context: Any) -> Any:
        handler = spec.handler
        if inspect.iscoroutinefunction(handler):
            return await handler(arguments, run_context)
        result = await asyncio.to_thread(handler, arguments, run_context)
        if inspect.isawaitable(result):
            return await result
        return result

    async def execute(
        self,
        spec: ToolSpec,
        call: ToolCall,
        *,
        run_context: Any | None = None,
    ) -> ToolResult:
        side_effect_capable = spec.permission in {
            "runtime_write",
            "memory_write",
            "workspace_write",
            "shell_exec",
            "network",
        }
        await self._emit(
            EngineEvent(
                event="tool.call.start",
                run_id=call.run_id,
                session_id=call.session_id,
                call_id=call.call_id,
                tool_name=spec.name,
                fields={"arguments_preview": str(call.arguments)[:200]},
            )
        )
        started = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._invoke(spec, call.arguments, run_context),
                timeout=float(spec.timeout_s),
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            data = raw if isinstance(raw, dict) else {"result": raw}
            result = ToolResult.success(
                call_id=call.call_id,
                tool_name=spec.name,
                data=data,
                latency_ms=latency_ms,
                metadata={
                    "timeout_s": spec.timeout_s,
                    "permission": spec.permission,
                    "tags": list(spec.tags or []),
                    "execution_flag": {
                        "called": True,
                        "executed": True,
                        "ok": True,
                        "permission": spec.permission,
                        "side_effect_capable": side_effect_capable,
                        "state_change_committed": side_effect_capable,
                    },
                },
            )
            await self._emit(
                EngineEvent(
                    event="tool.call.end",
                    run_id=call.run_id,
                    session_id=call.session_id,
                    call_id=call.call_id,
                    tool_name=spec.name,
                    fields={
                        "latency_ms": latency_ms,
                        "ok": True,
                    },
                )
            )
            return result
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - started) * 1000)
            result = ToolResult.failure(
                call_id=call.call_id,
                tool_name=spec.name,
                error_type="TimeoutError",
                message=f"tool timeout after {spec.timeout_s}s",
                latency_ms=latency_ms,
                metadata={
                    "timeout_s": spec.timeout_s,
                    "permission": spec.permission,
                    "tags": list(spec.tags or []),
                    "execution_flag": {
                        "called": True,
                        "executed": True,
                        "ok": False,
                        "permission": spec.permission,
                        "side_effect_capable": side_effect_capable,
                        "state_change_committed": False,
                    },
                },
            )
            await self._emit(
                EngineEvent(
                    event="tool.call.error",
                    run_id=call.run_id,
                    session_id=call.session_id,
                    call_id=call.call_id,
                    tool_name=spec.name,
                    level="error",
                    fields={"error_type": "TimeoutError", "latency_ms": latency_ms},
                )
            )
            return result
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            result = ToolResult.failure(
                call_id=call.call_id,
                tool_name=spec.name,
                error_type=type(exc).__name__,
                message=str(exc),
                latency_ms=latency_ms,
                metadata={
                    "permission": spec.permission,
                    "tags": list(spec.tags or []),
                    "execution_flag": {
                        "called": True,
                        "executed": True,
                        "ok": False,
                        "permission": spec.permission,
                        "side_effect_capable": side_effect_capable,
                        "state_change_committed": False,
                    },
                },
            )
            await self._emit(
                EngineEvent(
                    event="tool.call.error",
                    run_id=call.run_id,
                    session_id=call.session_id,
                    call_id=call.call_id,
                    tool_name=spec.name,
                    level="error",
                    fields={
                        "error_type": type(exc).__name__,
                        "latency_ms": latency_ms,
                    },
                )
            )
            return result
