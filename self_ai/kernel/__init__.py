"""Execution kernel scaffolding."""

from importlib import import_module
from typing import Any

__all__ = [
    "SelfAIKernel",
    "EngineEvent",
    "EngineLoop",
    "EngineState",
    "PromptRuntime",
    "ResultAggregator",
    "RunContext",
    "StateStore",
]


def __getattr__(name: str) -> Any:
    module_map = {
        "SelfAIKernel": ("kernel", "SelfAIKernel"),
        "EngineEvent": ("engine_events", "EngineEvent"),
        "EngineLoop": ("engine_loop", "EngineLoop"),
        "EngineState": ("engine_state", "EngineState"),
        "PromptRuntime": ("prompt_runtime", "PromptRuntime"),
        "ResultAggregator": ("result_aggregator", "ResultAggregator"),
        "RunContext": ("run_context", "RunContext"),
        "StateStore": ("state_store", "StateStore"),
    }
    if name not in module_map:
        raise AttributeError(f"module 'self_ai.kernel' has no attribute {name!r}")
    module_name, attr_name = module_map[name]
    module = import_module(f".{module_name}", __name__)
    return getattr(module, attr_name)
