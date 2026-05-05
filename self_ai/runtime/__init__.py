"""Tool runtime core package."""

from importlib import import_module
from typing import Any

__all__ = [
    "PermissionGuard",
    "ToolCall",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolRuntime",
    "ToolSpec",
]


def __getattr__(name: str) -> Any:
    module_map = {
        "PermissionGuard": ("permission_guard", "PermissionGuard"),
        "ToolCall": ("tool_call", "ToolCall"),
        "ToolExecutor": ("tool_executor", "ToolExecutor"),
        "ToolRegistry": ("tool_registry", "ToolRegistry"),
        "ToolResult": ("tool_result", "ToolResult"),
        "ToolRuntime": ("tool_runtime", "ToolRuntime"),
        "ToolSpec": ("tool_spec", "ToolSpec"),
    }
    if name not in module_map:
        raise AttributeError(f"module 'self_ai.runtime' has no attribute {name!r}")
    module_name, attr_name = module_map[name]
    module = import_module(f".{module_name}", __name__)
    return getattr(module, attr_name)
