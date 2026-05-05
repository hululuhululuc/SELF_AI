# coding=utf-8
"""Tool registry for Self AI ToolRuntime."""

from __future__ import annotations

from .tool_spec import ToolSpec


class DuplicateToolError(ValueError):
    """Raised when registering duplicate tool names."""


class MissingToolError(KeyError):
    """Raised when requested tool is missing."""


class ToolRegistry:
    """In-memory registry for tool specs."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise DuplicateToolError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise MissingToolError(f"tool not found: {name}")
        del self._tools[name]

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise MissingToolError(f"tool not found: {name}") from exc

    def list(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools)]
