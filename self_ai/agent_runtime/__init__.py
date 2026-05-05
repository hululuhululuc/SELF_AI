# coding=utf-8
"""Agent runtime package for multi-role execution and revision loop."""

from importlib import import_module
from typing import Any

__all__ = [
    "AgentLoop",
    "AgentPolicy",
    "AgentRuntime",
    "AgentSession",
    "MessageBus",
    "RevisionLoop",
    "TaskBoard",
]


def __getattr__(name: str) -> Any:
    module_map = {
        "AgentLoop": ("agent_loop", "AgentLoop"),
        "AgentPolicy": ("agent_policy", "AgentPolicy"),
        "AgentRuntime": ("runtime", "AgentRuntime"),
        "AgentSession": ("agent_session", "AgentSession"),
        "MessageBus": ("message_bus", "MessageBus"),
        "RevisionLoop": ("revision_loop", "RevisionLoop"),
        "TaskBoard": ("task_board", "TaskBoard"),
    }
    if name not in module_map:
        raise AttributeError(f"module 'self_ai.agent_runtime' has no attribute {name!r}")
    module_name, attr_name = module_map[name]
    module = import_module(f".{module_name}", __name__)
    return getattr(module, attr_name)

