# coding=utf-8
"""Self AI Kernel production entrypoint for EngineLoop."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from ..runtime.permission_guard import PermissionGuard
from ..runtime.tool_runtime import ToolRuntime
from ..observability import trace
from .engine_loop import EngineLoop
from .engine_state import EngineState
from .prompt_runtime import PromptRuntime
from .result_aggregator import ResultAggregator
from .run_context import RunContext
from .state_store import StateStore


class SelfAIKernel:
    """Kernel + ToolRuntime + EngineLoop orchestrator."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        settings: Any | None = None,
        permission_profile: dict[str, bool] | None = None,
        dependencies: dict[str, Any] | None = None,
        event_callback: Any | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.settings = settings
        self.permission_profile = permission_profile or {}
        self.dependencies = dependencies or {}
        self.event_callback = event_callback

        self.tool_runtime = ToolRuntime(
            permission_guard=PermissionGuard(self.permission_profile),
            event_callback=event_callback,
            dependencies={"project_root": self.project_root, **self.dependencies},
        )
        self.tool_runtime.register_default_tools()

        self.state_store = StateStore()
        self.prompt_runtime = PromptRuntime(
            max_chars=int(getattr(self.settings, "prompt_max_chars_mainloop", 5000) or 5000),
            cache_enabled=bool(getattr(self.settings, "prompt_cache_enabled", True)),
            cache_max_entries=int(getattr(self.settings, "prompt_cache_max_entries", 64) or 64),
            dynamic_budget_ratio=float(
                getattr(self.settings, "prompt_dynamic_budget_ratio", 0.35) or 0.35
            ),
            recent_errors=int(getattr(self.settings, "prompt_recent_errors", 3) or 3),
            recent_tool_results=int(
                getattr(self.settings, "prompt_recent_tool_results", 4) or 4
            ),
            selected_agents_max=int(
                getattr(self.settings, "prompt_selected_agents_max", 5) or 5
            ),
            tool_catalog_max=int(getattr(self.settings, "prompt_tool_catalog_max", 30) or 30),
        )
        self.result_aggregator = ResultAggregator()
        self.engine_loop = EngineLoop(
            tool_runtime=self.tool_runtime,
            state_store=self.state_store,
            prompt_runtime=self.prompt_runtime,
            result_aggregator=self.result_aggregator,
        )
        self._try_background_embedder_prewarm()

    def _try_background_embedder_prewarm(self) -> None:
        if not bool(getattr(self.settings, "embedder_prewarm", False)):
            return

        def _runner() -> None:
            try:
                from ..tools import prewarm_embedder

                prewarm_embedder()
            except Exception as exc:
                trace(
                    "embedder.warmup.error",
                    ok=False,
                    error=type(exc).__name__,
                    message=str(exc)[:220],
                    source="kernel.background",
                )

        try:
            t = threading.Thread(target=_runner, name="self-ai-embedder-prewarm", daemon=True)
            t.start()
            trace("embedder.warmup.thread.started", daemon=True)
        except Exception as exc:
            trace(
                "embedder.warmup.thread.error",
                error=type(exc).__name__,
                message=str(exc)[:220],
            )

    def new_run_context(
        self,
        *,
        run_id: str | None = None,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
        event_sink: Any | None = None,
    ) -> RunContext:
        return RunContext(
            project_root=self.project_root,
            run_id=run_id or str(uuid4()),
            session_id=session_id,
            settings=self.settings,
            permission_profile=self.permission_profile,
            tool_runtime=self.tool_runtime,
            state_store=self.state_store,
            event_sink=self.event_callback if event_sink is None else event_sink,
            metadata=metadata or {},
        )

    async def run_once(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        run_id: str | None = None,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = self.new_run_context(
            run_id=run_id,
            session_id=session_id,
            metadata=metadata,
        )
        result = await self.tool_runtime.execute_by_name(
            tool_name,
            arguments or {},
            run_id=context.run_id,
            session_id=context.session_id,
            metadata=context.metadata,
            run_context=context,
        )
        return result.model_dump(mode="json")

    async def run(
        self,
        input_text: str,
        *,
        session_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Production run path: Kernel.run -> EngineLoop."""
        context = self.new_run_context(session_id=session_id, metadata=metadata)
        state = EngineState.from_input(
            input_text,
            session_id=context.session_id,
            run_id=context.run_id,
            metadata=metadata,
        )
        final_state = await self.engine_loop.run(state, run_context=context)
        return final_state.to_result()

    def dry_run(self) -> dict[str, Any]:
        return {
            "tool_count": len(self.tool_runtime.registry.list()),
            "tools": [spec.name for spec in self.tool_runtime.registry.list()],
        }
