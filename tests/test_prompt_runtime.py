# coding=utf-8
"""Tests for PromptRuntime stage prompt building."""

from self_ai.kernel.prompt_runtime import PromptRuntime


def test_prompt_runtime_handles_missing_fields_safely() -> None:
    runtime = PromptRuntime(max_chars=1200)
    state = {}

    for stage in [
        "quick_answer",
        "analyze_task",
        "plan",
        "research",
        "role_agents",
        "review",
        "quality_gate_context",
        "final_response",
    ]:
        prompt = runtime.build_prompt(state, stage)
        assert isinstance(prompt, str)
        assert prompt


def test_prompt_runtime_final_response_prompt_is_length_limited() -> None:
    runtime = PromptRuntime(max_chars=1000)
    state = {
        "task": "x" * 2000,
        "workflow_decision": {"execution_mode": "code_focused"},
        "plan": {"steps": [{"step_id": "s1", "goal": "g"}]},
        "context_pack": {"evidence_items": [{"content": "c"}]},
        "agent_outputs": {"synthesizer": {"output": "y" * 4000}},
    }
    prompt = runtime.build_prompt(state, "final_response")
    assert len(prompt) <= 1000
    assert "Provide the final response" in prompt


def test_prompt_runtime_mainloop_prompt_has_layers_and_boundary() -> None:
    runtime = PromptRuntime(max_chars=4000)
    state = {
        "run_id": "r1",
        "session_id": "s1",
        "task": "Summarize Redis role",
        "errors": [{"type": "X", "message": "m"}],
        "selected_agents": [],
    }
    tools = [
        {
            "name": "workspace.file.list",
            "description": "List files",
            "permission": "workspace_read",
            "timeout_s": 5,
            "input_schema": {"required": ["path"]},
        }
    ]
    prompt = runtime.build_prompt(
        state,
        "mainloop_decide",
        context={"tools": tools, "turn_index": 1, "recent_tool_results": []},
    )
    assert "Tool Behavior Contract" in prompt
    assert "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__" in prompt
    assert "workspace.file.list" in prompt


def test_prompt_runtime_mainloop_prompt_exposes_cache_stats() -> None:
    runtime = PromptRuntime(
        max_chars=3000,
        cache_enabled=True,
        cache_max_entries=8,
        dynamic_budget_ratio=0.3,
        recent_errors=2,
        recent_tool_results=2,
    )
    state = {
        "run_id": "r1",
        "session_id": "s1",
        "task": "Explain Redis role",
        "errors": [{"type": "E1", "message": "m1", "stage": "mainloop"}],
    }
    context = {
        "tools": [
            {
                "name": "workspace.file.list",
                "description": "List files",
                "permission": "workspace_read",
                "timeout_s": 5,
                "input_schema": {"required": ["path"], "properties": {"path": {"type": "string"}}},
            }
        ],
        "turn_index": 1,
        "recent_tool_results": [],
    }

    runtime.build_prompt(state, "mainloop_decide", context=context)
    first_stats = runtime.consume_last_build_stats()
    assert first_stats["static_cache_hit"] is False
    assert first_stats["chars_total"] > 0
    assert first_stats["dynamic_ratio"] <= 0.8

    runtime.build_prompt(state, "mainloop_decide", context=context)
    second_stats = runtime.consume_last_build_stats()
    assert second_stats["static_cache_hit"] is True


def test_prompt_runtime_mainloop_prompt_keeps_task_in_dynamic_context_when_truncated() -> None:
    runtime = PromptRuntime(max_chars=900, dynamic_budget_ratio=0.5)
    state = {
        "run_id": "r2",
        "session_id": "s2",
        "task": "Create file artifacts/demo.txt with hello",
        "errors": [],
    }
    tools = []
    for idx in range(30):
        tools.append(
            {
                "name": f"tool.{idx}",
                "description": "x" * 120,
                "permission": "workspace_read",
                "timeout_s": 5,
                "input_schema": {
                    "required": ["path"],
                    "properties": {f"k{n}": {"type": "string"} for n in range(8)},
                },
            }
        )
    prompt = runtime.build_prompt(
        state,
        "mainloop_decide",
        context={"tools": tools, "turn_index": 1, "recent_tool_results": []},
    )
    assert "Task: Create file artifacts/demo.txt with hello" in prompt


def test_prompt_runtime_mainloop_prompt_includes_chat_recent_turn_summaries() -> None:
    runtime = PromptRuntime(max_chars=2200, dynamic_budget_ratio=0.5)
    state = {
        "run_id": "r3",
        "session_id": "s3",
        "task": "continue previous work",
        "metadata": {
            "chat_recent_turns": [
                {
                    "turn_id": 1,
                    "run_id": "r-old-1",
                    "task_preview": "create file a.py",
                    "response_preview": "created",
                    "retrieval_summary": "file creation done",
                    "execution": {"ok_tool_count": 1},
                }
            ],
            "chat_recent_turns_stats": {"requested_turns": 1, "kept_turns": 1},
            "chat_recent_turns_budget": {"max_chars_total": 1400, "used_chars": 260},
        },
    }
    prompt = runtime.build_prompt(
        state,
        "mainloop_decide",
        context={"tools": [], "turn_index": 2, "recent_tool_results": []},
    )
    assert "Recent chat turns stats" in prompt
    assert "file creation done" in prompt


def test_prompt_runtime_mainloop_injects_numbered_session_global_constraints_once() -> None:
    runtime = PromptRuntime(max_chars=6000, dynamic_budget_ratio=0.5)
    state = {
        "run_id": "r-gc",
        "session_id": "s-gc",
        "task": "create source file",
        "metadata": {
            "chat_global_constraints": [
                {"id": "gc-2", "content": "Use TypeScript for source files."},
                {"id": "gc-1", "content": "Prefer concise answers."},
            ]
        },
    }

    prompt = runtime.build_prompt(
        state,
        "mainloop_decide",
        context={"tools": [], "turn_index": 1, "recent_tool_results": []},
    )

    assert "Session Global Constraints:" in prompt
    assert "1. Use TypeScript for source files." in prompt
    assert "2. Prefer concise answers." in prompt
    assert "Active global constraints" not in prompt
    assert "constraint_maintenance" in prompt
    assert "authoritative over older retrieved chat memories" in prompt
    assert "do not infer an active session preference from older memory" in prompt


def test_prompt_runtime_mainloop_schema_includes_completion_outcome() -> None:
    runtime = PromptRuntime(max_chars=6000, dynamic_budget_ratio=0.5)
    prompt = runtime.build_prompt(
        {"run_id": "r-outcome", "session_id": "s-outcome", "task": "rename missing file"},
        "mainloop_decide",
        context={"tools": [], "turn_index": 1, "recent_tool_results": []},
    )

    assert '"outcome": "completed|blocked|failed"' in prompt
    assert "completion.outcome=completed only when" in prompt
    assert "completion.outcome=blocked or failed" in prompt


def test_prompt_runtime_mainloop_uses_expanded_recent_turn_window() -> None:
    runtime = PromptRuntime(max_chars=24000, dynamic_budget_ratio=0.7)
    turns = []
    for idx in range(1, 9):
        turns.append(
            {
                "turn_id": idx,
                "run_id": f"r-old-{idx}",
                "task_preview": f"task {idx}",
                "response_preview": f"response {idx}",
                "retrieval_summary": f"important historical detail {idx}",
                "execution": {"ok_tool_count": idx},
            }
        )
    state = {
        "run_id": "r4",
        "session_id": "s4",
        "task": "continue with broader memory",
        "metadata": {
            "chat_recent_turns": turns,
            "chat_recent_turns_stats": {"requested_turns": 8, "kept_turns": 8},
            "chat_recent_turns_budget": {"max_chars_total": 96000, "used_chars": 2048},
        },
    }

    prompt = runtime.build_prompt(
        state,
        "mainloop_decide",
        context={"tools": [], "turn_index": 9, "recent_tool_results": []},
    )

    assert "important historical detail 1" in prompt
    assert "important historical detail 8" in prompt
