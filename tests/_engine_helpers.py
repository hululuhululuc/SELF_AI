# coding=utf-8
"""Shared helpers for Kernel/EngineLoop test suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from self_ai.kernel import SelfAIKernel


class FakeRedisStore:
    """In-memory fake RedisStore-like object."""

    def __init__(self) -> None:
        self.states: list[tuple[str, dict[str, Any]]] = []
        self.traces: list[tuple[str, dict[str, Any]]] = []
        self.node_outputs: list[tuple[str, str, dict[str, Any]]] = []
        self.agent_outputs: list[tuple[str, str, dict[str, Any]]] = []

    def save_run_state(self, run_id: str, state: dict[str, Any]) -> None:
        self.states.append((run_id, state))

    def load_run_state(self, run_id: str) -> dict[str, Any] | None:
        for rid, state in reversed(self.states):
            if rid == run_id:
                return state
        return None

    def append_trace(self, run_id: str, trace: dict[str, Any]) -> None:
        self.traces.append((run_id, trace))

    def save_node_output(self, run_id: str, node_name: str, output: dict[str, Any]) -> None:
        self.node_outputs.append((run_id, node_name, output))

    def save_agent_output(self, run_id: str, agent_name: str, output: dict[str, Any]) -> None:
        self.agent_outputs.append((run_id, agent_name, output))


class FakeQdrantMemory:
    """Fake Qdrant memory: no external I/O."""

    async def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def upsert_evidence(self, **_kwargs: Any) -> bool:
        return True

    async def upsert_memory(self, **_kwargs: Any) -> bool:
        return True


class FakeGraphMemory:
    """Fake Neo4j memory: no external I/O."""

    def find_task_lineage(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def find_related_symbols(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def find_prior_issues(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def record_issue_fix(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


def make_route_model() -> Any:
    """Deterministic model stub for analyze/plan/review/final stages."""

    async def _route_model(prompt: str, category: str) -> dict[str, Any]:
        lower = prompt.lower()
        stage = str(category or "").strip().lower()

        if stage == "mainloop_decide":
            payload = {
                "type": "final_answer",
                "response": "final response",
                "execution_mode": "quick_answer",
                "workflow_decision": {
                    "execution_mode": "quick_answer",
                    "use_quick_answer": True,
                    "requires_context": False,
                },
                "completion": {
                    "is_complete": True,
                    "needs_side_effect_proof": False,
                    "needs_read_proof": False,
                    "evidence_refs": [],
                    "remaining_blockers": [],
                },
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

        if stage == "mainloop_goal_contract":
            payload = {
                "intent_kind": "analysis",
                "requires_side_effect": False,
                "requires_read_proof": False,
                "target_paths": [],
                "rationale": "mock goal contract",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

        if stage in {"mainloop_goal_contract_verify", "goal_contract_verify"}:
            payload = {
                "requires_side_effect": False,
                "confidence": 0.9,
                "reason": "mock vote",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

        if "goal-contract classifier" in lower:
            user_input = lower
            if "task:\n" in lower:
                user_input = lower.split("task:\n", 1)[1].strip()
            requires_side_effect = any(
                marker in user_input
                for marker in (
                    "create",
                    "write",
                    "edit",
                    "modify",
                    "update",
                    "delete",
                    "rename",
                    "migrate",
                    "创建",
                    "写入",
                    "修改",
                    "更新",
                    "删除",
                    "重命名",
                    "迁移",
                )
            )
            requires_read = (not requires_side_effect) and any(
                marker in user_input
                for marker in ("read", "inspect", "summarize", "读取", "查看", "总结", "检查")
            )
            payload = {
                "intent_kind": "state_change" if requires_side_effect else ("read_only" if requires_read else "analysis"),
                "requires_side_effect": requires_side_effect,
                "requires_read_proof": requires_read,
                "target_paths": [],
                "rationale": "mock goal contract",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

        if "does completing the user task require mutating workspace state" in lower:
            user_input = lower
            if "task:\n" in lower:
                user_input = lower.split("task:\n", 1)[1].strip()
            requires_side_effect = any(
                marker in user_input
                for marker in (
                    "create",
                    "write",
                    "edit",
                    "modify",
                    "update",
                    "delete",
                    "rename",
                    "migrate",
                    "创建",
                    "写入",
                    "修改",
                    "更新",
                    "删除",
                    "重命名",
                    "迁移",
                )
            )
            payload = {
                "requires_side_effect": requires_side_effect,
                "confidence": 0.9,
                "reason": "mock vote",
            }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

        if "classify the user task" in lower:
            user_input = lower
            if "user input:" in lower:
                user_input = lower.split("user input:", 1)[1].strip()
            elif "\ntask:\n" in lower:
                user_input = lower.split("\ntask:\n", 1)[1].strip()

            if "architecture" in user_input or "planning" in user_input:
                payload = {
                    "intent": "architecture_design",
                    "domain": "software",
                    "requires_code": False,
                    "requires_repo_context": False,
                    "requires_research": True,
                    "requires_planning": True,
                    "requires_review": True,
                    "expected_artifact": "design_doc",
                    "complexity": "high",
                    "risk_level": "high",
                    "confidence": 0.9,
                    "rationale": "architecture task",
                }
            elif "document" in user_input or "write a short project" in user_input:
                payload = {
                    "intent": "document_writing",
                    "domain": "general",
                    "requires_code": False,
                    "requires_repo_context": False,
                    "requires_research": False,
                    "requires_planning": False,
                    "requires_review": True,
                    "expected_artifact": "answer",
                    "complexity": "low",
                    "risk_level": "low",
                    "confidence": 0.9,
                    "rationale": "document writing task",
                }
            elif "research summary" in user_input or "summarize" in user_input:
                payload = {
                    "intent": "research_summary",
                    "domain": "general",
                    "requires_code": False,
                    "requires_repo_context": False,
                    "requires_research": True,
                    "requires_planning": False,
                    "requires_review": True,
                    "expected_artifact": "research_report",
                    "complexity": "medium",
                    "risk_level": "medium",
                    "confidence": 0.9,
                    "rationale": "research summary task",
                }
            elif "code" in user_input or "python" in user_input or "function" in user_input:
                payload = {
                    "intent": "code_generation",
                    "domain": "software",
                    "requires_code": True,
                    "requires_repo_context": False,
                    "requires_research": False,
                    "requires_planning": False,
                    "requires_review": True,
                    "expected_artifact": "code_patch",
                    "complexity": "medium",
                    "risk_level": "medium",
                    "confidence": 0.9,
                    "rationale": "code generation task",
                }
            else:
                payload = {
                    "intent": "general_qa",
                    "domain": "general",
                    "requires_code": False,
                    "requires_repo_context": False,
                    "requires_research": False,
                    "requires_planning": False,
                    "requires_review": False,
                    "expected_artifact": "answer",
                    "complexity": "low",
                    "risk_level": "low",
                    "confidence": 0.9,
                    "rationale": "quick answer task",
                }
            return {"model": "mock-fast", "response": json.dumps(payload, ensure_ascii=False)}

        if "return json only with keys: task_summary, execution_mode, rationale, steps" in lower:
            payload = {
                "task_summary": "planned task",
                "execution_mode": "architecture_design",
                "rationale": "structured plan",
                "steps": [
                    {
                        "step_id": "s1",
                        "goal": "analyze",
                        "agent_name": "architect",
                        "depends_on": [],
                        "expected_output": "analysis",
                        "done_criteria": "scope identified",
                    },
                    {
                        "step_id": "s2",
                        "goal": "implement",
                        "agent_name": "coder",
                        "depends_on": ["s1"],
                        "expected_output": "draft",
                        "done_criteria": "draft complete",
                    },
                ],
            }
            return {"model": "mock-reasoning", "response": json.dumps(payload, ensure_ascii=False)}

        if "storage-driven reviewer" in lower:
            payload = {
                "pass_review": True,
                "severity": "medium",
                "major_issues": [],
                "minor_issues": ["minor formatting"],
                "missing_requirements": [],
                "requires_revision": False,
                "revision_target_agent": None,
                "revision_instruction": "",
                "evidence_refs": [],
                "storage_evidence": [],
                "rationale": "looks good",
            }
            return {"model": "mock-review", "response": json.dumps(payload, ensure_ascii=False)}

        return {"model": f"mock-{category}", "response": json.dumps({"type": "final_answer", "response": "final response"}, ensure_ascii=False)}

    return _route_model


def build_test_kernel(tmp_path: Path, *, permission_profile: dict[str, bool] | None = None) -> tuple[SelfAIKernel, FakeRedisStore]:
    """Create a fully-isolated Kernel for Step-2 tests."""
    redis_store = FakeRedisStore()
    kernel = SelfAIKernel(
        project_root=tmp_path,
        permission_profile=permission_profile
        or {
            "read_only": True,
            "runtime_write": True,
            "memory_read": True,
            "memory_write": False,
            "workspace_read": True,
            "workspace_write": False,
            "shell_exec": False,
            "network": False,
            "model_call": True,
        },
        dependencies={
            "route_model_func": make_route_model(),
            "redis_store": redis_store,
            "qdrant_memory": FakeQdrantMemory(),
            "graph_memory": FakeGraphMemory(),
            "web_search_func": lambda _query, _max_results: [],
            "project_root": str(tmp_path),
        },
    )
    return kernel, redis_store
