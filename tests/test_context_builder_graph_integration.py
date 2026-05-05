# coding=utf-8
"""Phase 4.5 tests for ContextBuilder graph-path integration."""

from self_ai.graph.graph_models import GraphEdge, GraphNode, GraphPath
from self_ai.retrieval.context_builder import build_context_pack
from self_ai.schemas import EvidenceItem, RetrievalPolicy, TaskIntent, TaskProfile


def _profile() -> TaskProfile:
    return TaskProfile(intent=TaskIntent.DEBUGGING)


def test_no_graph_paths_keeps_previous_behavior() -> None:
    pack = build_context_pack(
        query="debug value error",
        profile=_profile(),
        retrieval_policy=RetrievalPolicy(max_context_chars=300),
        evidence_items=[
            EvidenceItem(content="high", source_type="code", score=0.9, chunk_id="a"),
            EvidenceItem(content="low", source_type="code", score=0.2, chunk_id="b"),
        ],
        graph_paths=None,
    )
    assert pack["selected_evidence_count"] == 2
    assert pack["evidence"][0]["content"] == "high"
    assert pack["graph_summary"]["path_count"] == 0
    assert pack["graph_paths"] == []


def test_graph_paths_dedup_and_sort() -> None:
    path_low = GraphPath(
        nodes=[GraphNode(id="f:a", label="Function"), GraphNode(id="f:b", label="Function")],
        edges=[GraphEdge(source_id="f:a", target_id="f:b", relation="CALLS")],
        metadata={"score": 0.2},
    )
    path_high = GraphPath(
        nodes=[GraphNode(id="f:a", label="Function"), GraphNode(id="f:b", label="Function")],
        edges=[GraphEdge(source_id="f:a", target_id="f:b", relation="CALLS")],
        metadata={"score": 0.9},
    )
    path_other = GraphPath(
        nodes=[GraphNode(id="issue:1", label="Issue"), GraphNode(id="fix:1", label="Fix")],
        edges=[GraphEdge(source_id="issue:1", target_id="fix:1", relation="FIXED_BY")],
        metadata={"score": 0.7},
    )
    pack = build_context_pack(
        query="debug",
        profile=_profile(),
        retrieval_policy=RetrievalPolicy(max_context_chars=500),
        evidence_items=[EvidenceItem(content="x", source_type="code", score=0.5, chunk_id="x")],
        graph_paths=[path_low, path_high, path_other],
    )
    assert pack["graph_summary"]["path_count"] == 2
    assert pack["graph_paths"][0]["metadata"]["score"] == 0.9


def test_graph_paths_grouped_by_relation_type() -> None:
    pack = build_context_pack(
        query="debug",
        profile=_profile(),
        retrieval_policy=RetrievalPolicy(max_context_chars=500),
        evidence_items=[EvidenceItem(content="x", source_type="code", score=0.5, chunk_id="x")],
        graph_paths=[
            GraphPath(
                nodes=[GraphNode(id="f:a", label="Function"), GraphNode(id="f:b", label="Function")],
                edges=[GraphEdge(source_id="f:a", target_id="f:b", relation="CALLS")],
            ),
            GraphPath(
                nodes=[GraphNode(id="issue:1", label="Issue"), GraphNode(id="fix:1", label="Fix")],
                edges=[GraphEdge(source_id="issue:1", target_id="fix:1", relation="FIXED_BY")],
            ),
        ],
    )
    counts = pack["graph_summary"]["relation_type_counts"]
    assert counts["CALLS"] == 1
    assert counts["FIXED_BY"] == 1


def test_graph_context_preview_is_truncated() -> None:
    pack = build_context_pack(
        query="debug",
        profile=_profile(),
        retrieval_policy=RetrievalPolicy(max_context_chars=150),
        evidence_items=[EvidenceItem(content="x", source_type="code", score=0.5, chunk_id="x")],
        graph_paths=[
            GraphPath(
                nodes=[GraphNode(id="f:a", label="Function"), GraphNode(id="f:b", label="Function")],
                edges=[GraphEdge(source_id="f:a", target_id="f:b", relation="CALLS")],
            ),
            GraphPath(
                nodes=[GraphNode(id="f:b", label="Function"), GraphNode(id="f:c", label="Function")],
                edges=[GraphEdge(source_id="f:b", target_id="f:c", relation="CALLS")],
            ),
        ],
    )
    assert len(pack["graph_context_preview"]) <= 30  # 150 * 20%


def test_graph_summary_counts_are_correct() -> None:
    pack = build_context_pack(
        query="debug",
        profile=_profile(),
        retrieval_policy=RetrievalPolicy(max_context_chars=500),
        evidence_items=[EvidenceItem(content="x", source_type="code", score=0.5, chunk_id="x")],
        graph_paths=[
            GraphPath(
                nodes=[GraphNode(id="f:a", label="Function"), GraphNode(id="f:b", label="Function")],
                edges=[GraphEdge(source_id="f:a", target_id="f:b", relation="CALLS")],
            ),
            GraphPath(
                nodes=[GraphNode(id="f:b", label="Function"), GraphNode(id="f:c", label="Function")],
                edges=[GraphEdge(source_id="f:b", target_id="f:c", relation="RELATED_TO")],
            ),
        ],
    )
    summary = pack["graph_summary"]
    assert summary["path_count"] == 2
    assert summary["node_count"] == 3
    assert summary["relation_count"] == 2


def test_graph_paths_do_not_store_large_raw_content() -> None:
    huge = "A" * 5000
    path = GraphPath(
        nodes=[GraphNode(id="n1", label="Function", properties={"content": huge})],
        edges=[],
        metadata={"summary": huge},
    )
    pack = build_context_pack(
        query="debug",
        profile=_profile(),
        retrieval_policy=RetrievalPolicy(max_context_chars=500),
        evidence_items=[EvidenceItem(content="x", source_type="code", score=0.5, chunk_id="x")],
        graph_paths=[path],
    )
    assert huge not in str(pack["graph_paths"])


def test_graph_context_does_not_affect_evidence_ranking() -> None:
    pack = build_context_pack(
        query="debug",
        profile=_profile(),
        retrieval_policy=RetrievalPolicy(max_context_chars=500),
        evidence_items=[
            EvidenceItem(content="best-evidence", source_type="code", score=0.95, chunk_id="h"),
            EvidenceItem(content="worse-evidence", source_type="code", score=0.50, chunk_id="l"),
        ],
        graph_paths=[
            GraphPath(
                nodes=[GraphNode(id="issue:1", label="Issue"), GraphNode(id="fix:1", label="Fix")],
                edges=[GraphEdge(source_id="issue:1", target_id="fix:1", relation="FIXED_BY")],
            )
        ],
    )
    assert pack["evidence"][0]["content"] == "best-evidence"
