# coding=utf-8
"""Tests for Phase 4.5 GraphQueryAdapter (read-only graph integration)."""

from self_ai.graph.graph_models import GraphEdge, GraphNode, GraphPath
from self_ai.retrieval.graph_query_adapter import GraphQueryAdapter, GraphQueryPlan
from self_ai.schemas import EvidenceItem, RetrievalPolicy, TaskIntent, TaskProfile


class FakeGraphMemory:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.raise_in: set[str] = set()

    def find_related_symbols(self, symbol_name: str, limit: int = 10) -> list[GraphPath]:
        self.calls.append(("find_related_symbols", symbol_name))
        if "find_related_symbols" in self.raise_in:
            raise RuntimeError("boom")
        return [
            GraphPath(
                nodes=[
                    GraphNode(id=f"f:{symbol_name}", label="Function"),
                    GraphNode(id=f"g:{symbol_name}", label="Function"),
                ],
                edges=[GraphEdge(source_id=f"f:{symbol_name}", target_id=f"g:{symbol_name}", relation="CALLS")],
                metadata={"score": 0.9},
            )
            for _ in range(limit)
        ]

    def find_prior_issues(self, query: str, limit: int = 10) -> list[GraphPath]:
        self.calls.append(("find_prior_issues", query))
        if "find_prior_issues" in self.raise_in:
            raise RuntimeError("boom")
        return [
            GraphPath(
                nodes=[GraphNode(id="issue:1", label="Issue"), GraphNode(id="fix:1", label="Fix")],
                edges=[GraphEdge(source_id="issue:1", target_id="fix:1", relation="FIXED_BY")],
                metadata={"score": 0.7},
            )
        ][:limit]

    def find_task_lineage(self, run_id: str, limit: int = 20) -> GraphPath:
        self.calls.append(("find_task_lineage", run_id))
        if "find_task_lineage" in self.raise_in:
            raise RuntimeError("boom")
        return GraphPath(
            nodes=[GraphNode(id=f"task:{run_id}", label="Task"), GraphNode(id=f"profile:{run_id}", label="TaskProfile")],
            edges=[GraphEdge(source_id=f"task:{run_id}", target_id=f"profile:{run_id}", relation="HAS_PROFILE")],
            metadata={"run_id": run_id, "score": 0.6},
        )


def _profile(intent: TaskIntent) -> TaskProfile:
    profile = TaskProfile(intent=intent)
    profile.identity.run_id = "run-graph"
    return profile


def test_mapping_for_debugging_code_review_code_modification() -> None:
    adapter = GraphQueryAdapter(FakeGraphMemory(), enabled=True)
    for intent in [TaskIntent.DEBUGGING, TaskIntent.CODE_REVIEW, TaskIntent.CODE_MODIFICATION]:
        plan = adapter.plan_graph_queries(
            query="main.py ValueError traceback",
            task_profile=_profile(intent),
            retrieval_policy=RetrievalPolicy(use_graph=True),
            evidence_items=[],
        )
        assert plan.enabled is True
        assert plan.query_types == ["find_related_symbols", "find_prior_issues", "find_task_lineage"]


def test_mapping_for_architecture_and_planning() -> None:
    adapter = GraphQueryAdapter(FakeGraphMemory(), enabled=True)
    for intent in [TaskIntent.ARCHITECTURE_DESIGN, TaskIntent.PLANNING]:
        plan = adapter.plan_graph_queries(
            query="refactor workflow topology",
            task_profile=_profile(intent),
            retrieval_policy=RetrievalPolicy(use_graph=True),
            evidence_items=[],
        )
        assert plan.enabled is True
        assert plan.query_types == ["find_task_lineage", "find_related_symbols"]


def test_research_summary_requires_use_graph_true() -> None:
    adapter = GraphQueryAdapter(FakeGraphMemory(), enabled=True)
    profile = _profile(TaskIntent.RESEARCH_SUMMARY)
    enabled_plan = adapter.plan_graph_queries(
        query="summarize journal requirements",
        task_profile=profile,
        retrieval_policy=RetrievalPolicy(use_graph=True),
        evidence_items=[],
    )
    assert enabled_plan.enabled is True
    assert enabled_plan.query_types == ["find_task_lineage"]

    disabled_plan = adapter.plan_graph_queries(
        query="summarize journal requirements",
        task_profile=profile,
        retrieval_policy=RetrievalPolicy(use_graph=False),
        evidence_items=[],
    )
    assert disabled_plan.enabled is False


def test_document_writing_and_general_default_do_not_trigger() -> None:
    adapter = GraphQueryAdapter(FakeGraphMemory(), enabled=True)
    for intent in [TaskIntent.DOCUMENT_WRITING, TaskIntent.GENERAL_QA]:
        plan = adapter.plan_graph_queries(
            query="write polished abstract in English",
            task_profile=_profile(intent),
            retrieval_policy=RetrievalPolicy(use_graph=True),
            evidence_items=[],
        )
        assert plan.enabled is False
        assert plan.query_types == []


def test_symbol_extraction_from_evidence_item_symbol() -> None:
    adapter = GraphQueryAdapter(FakeGraphMemory(), enabled=True)
    plan = adapter.plan_graph_queries(
        query="please debug this",
        task_profile=_profile(TaskIntent.CODE_GENERATION),
        retrieval_policy=RetrievalPolicy(use_graph=True),
        evidence_items=[EvidenceItem(content="x", source_type="code", symbol="merge_k_lists")],
    )
    assert "merge_k_lists" in plan.symbols
    assert "find_related_symbols" in plan.query_types


def test_symbol_extraction_from_query_code_like_token() -> None:
    adapter = GraphQueryAdapter(FakeGraphMemory(), enabled=True)
    plan = adapter.plan_graph_queries(
        query="please optimize merge_k_lists in linked list implementation",
        task_profile=_profile(TaskIntent.CODE_GENERATION),
        retrieval_policy=RetrievalPolicy(use_graph=True),
        evidence_items=[],
    )
    assert "merge_k_lists" in plan.symbols
    assert "find_related_symbols" in plan.query_types


def test_enabled_false_does_not_call_graph_memory() -> None:
    fake = FakeGraphMemory()
    adapter = GraphQueryAdapter(fake, enabled=False)
    plan = adapter.plan_graph_queries(
        query="debug",
        task_profile=_profile(TaskIntent.DEBUGGING),
        retrieval_policy=RetrievalPolicy(use_graph=True),
        evidence_items=[],
    )
    assert plan.enabled is False
    assert adapter.run_graph_queries(plan) == []
    assert fake.calls == []


def test_neo4j_exception_returns_empty_paths() -> None:
    fake = FakeGraphMemory()
    fake.raise_in.add("find_related_symbols")
    adapter = GraphQueryAdapter(fake, enabled=True)
    plan = GraphQueryPlan(
        enabled=True,
        query_types=["find_related_symbols"],
        symbols=["merge_k_lists"],
        max_paths=5,
    )
    paths = adapter.run_graph_queries(plan)
    assert paths == []


def test_max_paths_limit_is_enforced() -> None:
    fake = FakeGraphMemory()
    adapter = GraphQueryAdapter(fake, enabled=True, max_paths=2)
    plan = GraphQueryPlan(
        enabled=True,
        query_types=["find_related_symbols"],
        symbols=["merge_k_lists"],
        max_paths=2,
    )
    paths = adapter.run_graph_queries(plan)
    assert len(paths) <= 2
