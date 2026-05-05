# coding=utf-8
"""State definition for Self AI LangGraph workflows.

This module defines the state structure and utilities for hierarchical memory.
"""

from collections import deque
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    """Workflow state for Self AI.

    Attributes:
        messages: Short-term shared messages, annotated for addition.
        task_queue: In-memory task queue.
        private: Per-agent private state.
        long_term: Persistent long-term state via checkpointer.
        input: Input query or PRD.
        task: Active task being processed.
        run_id: Unique run identifier for traceability.
        session_id: Session identifier for continuity.
        task_profile: Structured profile produced by task analyzer.
        retrieval_policy: Retrieval policy snapshot for current run.
        memory_write_policy: Memory write policy snapshot for current run.
        workflow_decision: Workflow decision snapshot for current run.
        plan: Structured execution plan for complex tasks.
        context_pack: First-class compact context package for implement stage.
        selected_agents: Selected role-agent sequence for current run.
        research_results: Retrieval outputs from GraphRAG+.
        agent_outputs: Collected agent outputs for later stages.
        review_reports: Collected review reports for later stages.
        quality_gate: Structured quality-gate decision for current run.
        model: Final selected model for implementation stage.
        response: Final generated response text.
        errors: Structured error list.
    """

    messages: list[dict[str, Any]]
    task_queue: deque[str]
    private: dict[str, Any]
    long_term: dict[str, Any]
    input: str
    task: str
    run_id: str
    session_id: str
    task_profile: dict[str, Any]
    retrieval_policy: dict[str, Any]
    memory_write_policy: dict[str, Any]
    workflow_decision: dict[str, Any]
    plan: dict[str, Any]
    context_pack: dict[str, Any]
    selected_agents: list[str]
    research_results: list[dict[str, Any]]
    agent_outputs: dict[str, Any]
    review_reports: list[dict[str, Any]]
    quality_gate: dict[str, Any]
    model: str
    response: str
    errors: list[dict[str, Any]]


def cap_messages(state: State, max_messages: int = 50) -> State:
    """Cap shared messages to prevent bloat and ensure low latency.

    Args:
        state: Current workflow state.
        max_messages: Maximum messages to retain (default: 50).

    Returns:
        Updated state with capped messages.
    """
    if len(state["messages"]) > max_messages:
        state["messages"] = state["messages"][-max_messages:]
    return state


# Checkpointer for persistence (integrated in main.py)
checkpointer: MemorySaver = MemorySaver()
