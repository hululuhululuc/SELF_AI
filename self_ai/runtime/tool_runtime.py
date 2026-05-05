# coding=utf-8
"""Unified tool runtime wiring registry + permission + executor."""

from __future__ import annotations

from typing import Any

from ..kernel.engine_events import EngineEvent
from .permission_guard import PermissionGuard
from .tool_call import ToolCall
from .tool_executor import EventCallback, ToolExecutor
from .tool_registry import MissingToolError, ToolRegistry
from .tool_result import ToolResult


class ToolRuntime:
    """Coordinates tool lookup, permission checks, and execution."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        permission_guard: PermissionGuard | None = None,
        event_callback: EventCallback | None = None,
        dependencies: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.permission_guard = permission_guard or PermissionGuard()
        self.executor = executor or ToolExecutor(event_callback=event_callback)
        self.event_callback = event_callback
        self.dependencies = dependencies or {}

    async def _emit(self, event: EngineEvent) -> None:
        if self.event_callback is None:
            return
        try:
            maybe = self.event_callback(event)
            if hasattr(maybe, "__await__"):
                await maybe
        except Exception:
            return

    def register_default_tools(self, dependencies: dict[str, Any] | None = None) -> None:
        """Register phase-7R step-1 default tools via adapters."""
        deps = dict(self.dependencies)
        if dependencies:
            deps.update(dependencies)

        from ..adapters.agent_tools import register_agent_tools
        from ..adapters.model_tool import register_model_tools
        from ..adapters.neo4j_tools import register_neo4j_tools
        from ..adapters.qdrant_tools import register_qdrant_tools
        from ..adapters.redis_tools import register_redis_tools
        from ..adapters.review_tools import register_review_tools
        from ..adapters.web_tools import register_web_tools
        from ..adapters.workspace_tools import register_workspace_tools

        register_model_tools(self.registry, route_model_func=deps.get("route_model_func"))
        register_redis_tools(self.registry, redis_store=deps.get("redis_store"))
        register_qdrant_tools(self.registry, qdrant_memory=deps.get("qdrant_memory"))
        register_neo4j_tools(self.registry, graph_memory=deps.get("graph_memory"))
        register_web_tools(self.registry, web_search_func=deps.get("web_search_func"))
        register_workspace_tools(
            self.registry,
            project_root=deps.get("project_root"),
            shell_enabled=bool(deps.get("shell_enabled", False)),
            shell_whitelist=deps.get("shell_whitelist"),
            shell_denylist=deps.get("shell_denylist"),
            max_file_bytes=int(deps.get("max_file_bytes", 1_048_576)),
        )
        register_agent_tools(
            self.registry,
            agent_map=deps.get("agent_map"),
            route_model_func=deps.get("route_model_func"),
        )
        register_review_tools(
            self.registry,
            reviewer=deps.get("reviewer"),
            route_model_func=deps.get("route_model_func"),
        )

    async def execute(self, call: ToolCall, *, run_context: Any | None = None) -> ToolResult:
        """Execute one ToolCall with permission checks and safe failure."""
        try:
            spec = self.registry.get(call.tool_name)
        except MissingToolError as exc:
            return ToolResult.failure(
                call_id=call.call_id,
                tool_name=call.tool_name,
                error_type="MissingToolError",
                message=str(exc),
            )

        allowed = self.permission_guard.can(spec.permission, run_context=run_context)
        if not allowed:
            side_effect_capable = spec.permission in {
                "runtime_write",
                "memory_write",
                "workspace_write",
                "shell_exec",
                "network",
            }
            await self._emit(
                EngineEvent(
                    event="tool.call.error",
                    run_id=call.run_id,
                    session_id=call.session_id,
                    call_id=call.call_id,
                    tool_name=spec.name,
                    level="warn",
                    fields={"error_type": "PermissionDenied", "permission": spec.permission},
                )
            )
            return ToolResult.failure(
                call_id=call.call_id,
                tool_name=spec.name,
                error_type="PermissionDenied",
                message=self.permission_guard.deny_reason(spec.permission),
                metadata={
                    "permission": spec.permission,
                    "execution_flag": {
                        "called": True,
                        "executed": False,
                        "ok": False,
                        "permission": spec.permission,
                        "side_effect_capable": side_effect_capable,
                        "state_change_committed": False,
                        "permission_denied": True,
                    },
                },
            )

        return await self.executor.execute(spec, call, run_context=run_context)

    async def execute_by_name(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        run_id: str = "",
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
        run_context: Any | None = None,
    ) -> ToolResult:
        call = ToolCall(
            run_id=run_id,
            session_id=session_id,
            tool_name=name,
            arguments=arguments or {},
            metadata=metadata or {},
        )
        return await self.execute(call, run_context=run_context)
