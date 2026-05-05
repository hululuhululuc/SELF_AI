# coding=utf-8
"""Package initializer for Self AI."""

from importlib import import_module
from typing import Any, Final

__version__: Final[str] = "0.1.0"

__all__ = [
    "Settings",
    "run_autonomy_workflow",
    "run_tool_once",
    "get_chat_manifest",
    "replay_chat_turns",
    "load_recent_chat_turns",
    "route_model",
    "State",
    "graphrag_plus",
    "TaskProfile",
    "RunIdentity",
    "RetrievalPolicy",
    "MemoryWritePolicy",
    "analyze_task",
    "WorkflowDecision",
    "decide_workflow",
]


def __getattr__(name: str) -> Any:
    """Lazily import public symbols to avoid heavy side effects at import time."""
    module_map = {
        "Settings": ("config", "Settings"),
        "run_autonomy_workflow": ("main", "run_autonomy_workflow"),
        "run_tool_once": ("main", "run_tool_once"),
        "get_chat_manifest": ("main", "get_chat_manifest"),
        "replay_chat_turns": ("main", "replay_chat_turns"),
        "load_recent_chat_turns": ("main", "load_recent_chat_turns"),
        "route_model": ("router", "route_model"),
        "State": ("state", "State"),
        "graphrag_plus": ("tools", "graphrag_plus"),
        "TaskProfile": ("schemas", "TaskProfile"),
        "RunIdentity": ("schemas", "RunIdentity"),
        "RetrievalPolicy": ("schemas", "RetrievalPolicy"),
        "MemoryWritePolicy": ("schemas", "MemoryWritePolicy"),
        "analyze_task": ("task_analyzer", "analyze_task"),
        "WorkflowDecision": ("workflow_policy", "WorkflowDecision"),
        "decide_workflow": ("workflow_policy", "decide_workflow"),
    }
    if name not in module_map:
        raise AttributeError(f"module 'self_ai' has no attribute {name!r}")

    module_name, attr_name = module_map[name]
    module = import_module(f".{module_name}", __name__)
    return getattr(module, attr_name)
