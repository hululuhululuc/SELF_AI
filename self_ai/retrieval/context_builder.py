# coding=utf-8
"""Context building from EvidenceItem hits for Phase 3."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..graph.graph_models import GraphPath
from ..schemas import EvidenceItem, RetrievalPolicy, TaskProfile


class ContextBuilder:
    """Build compact context packs from retrieval evidence."""

    def build(
        self,
        *,
        query: str,
        profile: TaskProfile,
        retrieval_policy: RetrievalPolicy,
        evidence_items: list[EvidenceItem | dict[str, Any]],
        graph_paths: list[GraphPath | dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return build_context_pack(
            query=query,
            profile=profile,
            retrieval_policy=retrieval_policy,
            evidence_items=evidence_items,
            graph_paths=graph_paths,
        )


def _normalize_items(items: list[EvidenceItem | dict[str, Any]]) -> list[EvidenceItem]:
    normalized: list[EvidenceItem] = []
    for item in items:
        if isinstance(item, EvidenceItem):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            normalized.append(EvidenceItem(**item))
    return normalized


def _dedup_key(item: EvidenceItem) -> str:
    marker = item.chunk_id or item.url or item.path or item.content[:120]
    return f"{item.source_type}|{marker}"


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 3:
        return text[:max_chars], True
    return text[: max_chars - 3] + "...", True


def _normalize_graph_paths(
    graph_paths: list[GraphPath | dict[str, Any]] | None,
) -> list[GraphPath]:
    if not graph_paths:
        return []
    normalized: list[GraphPath] = []
    for item in graph_paths:
        if isinstance(item, GraphPath):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            try:
                normalized.append(GraphPath(**item))
            except Exception:
                continue
    return normalized


def _graph_path_signature(path: GraphPath) -> str:
    node_key = "|".join(node.id for node in path.nodes)
    rel_key = "|".join(edge.relation for edge in path.edges)
    endpoint = ""
    if path.nodes:
        endpoint = f"{path.nodes[0].id}->{path.nodes[-1].id}"
    return f"{node_key}::{rel_key}::{endpoint}"


def _graph_path_score(path: GraphPath) -> float:
    metadata = path.metadata if isinstance(path.metadata, dict) else {}
    raw = metadata.get("score")
    try:
        return float(raw) if raw is not None else 0.0
    except Exception:
        return 0.0


def _relation_bucket(relation: str) -> str:
    known = {
        "CALLS",
        "CONTAINS",
        "FIXED_BY",
        "TESTS",
        "USED_EVIDENCE",
        "HAS_REVIEW",
        "HAS_OUTPUT",
        "RELATED_TO",
    }
    rel = (relation or "").strip().upper()
    return rel if rel in known else "OTHER"


def _compact_graph_path(path: GraphPath) -> dict[str, Any]:
    metadata = path.metadata if isinstance(path.metadata, dict) else {}
    return {
        "nodes": [{"id": node.id, "label": node.label} for node in path.nodes],
        "edges": [
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
            }
            for edge in path.edges
        ],
        "metadata": {
            "score": metadata.get("score"),
            "run_id": metadata.get("run_id"),
            "symbol_name": metadata.get("symbol_name"),
            "query": metadata.get("query"),
            "result_count": metadata.get("result_count"),
        },
    }


def _graph_preview_line(path: GraphPath) -> str:
    start = path.nodes[0].label if path.nodes else "Node"
    end = path.nodes[-1].label if path.nodes else "Node"
    rels = " -> ".join(edge.relation for edge in path.edges) or "RELATED_TO"
    return f"[graph] {start} {rels} {end}"


def _build_graph_context(
    graph_paths: list[GraphPath | dict[str, Any]] | None,
    max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    normalized = _normalize_graph_paths(graph_paths)
    if not normalized:
        return (
            [],
            {
                "path_count": 0,
                "node_count": 0,
                "relation_count": 0,
                "relation_type_counts": {},
            },
            "",
        )

    deduped: dict[str, GraphPath] = {}
    for path in normalized:
        key = _graph_path_signature(path)
        prev = deduped.get(key)
        if prev is None or _graph_path_score(path) > _graph_path_score(prev):
            deduped[key] = path

    ranked = sorted(
        deduped.values(),
        key=lambda p: (_graph_path_score(p), -len(p.edges)),
        reverse=True,
    )

    relation_type_counts: dict[str, int] = defaultdict(int)
    unique_nodes: set[str] = set()
    relation_count = 0
    compact_paths: list[dict[str, Any]] = []

    graph_budget = min(max(0, int(max_chars * 0.2)), 1200)
    preview_lines: list[str] = []
    used_chars = 0

    for path in ranked:
        compact_paths.append(_compact_graph_path(path))
        for node in path.nodes:
            unique_nodes.add(node.id)
        for edge in path.edges:
            relation_type_counts[_relation_bucket(edge.relation)] += 1
            relation_count += 1

        if graph_budget > 0:
            line = _graph_preview_line(path)
            pending = len(line) + 1
            if used_chars + pending <= graph_budget:
                preview_lines.append(line)
                used_chars += pending
            else:
                break

    preview = "\n".join(preview_lines)
    preview, _ = _truncate(preview, graph_budget if graph_budget > 0 else 0)
    return (
        compact_paths,
        {
            "path_count": len(compact_paths),
            "node_count": len(unique_nodes),
            "relation_count": relation_count,
            "relation_type_counts": dict(relation_type_counts),
        },
        preview,
    )


def build_context_pack(
    *,
    query: str,
    profile: TaskProfile,
    retrieval_policy: RetrievalPolicy,
    evidence_items: list[EvidenceItem | dict[str, Any]],
    graph_paths: list[GraphPath | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a grouped and truncated context pack from evidence items."""
    max_chars = int(retrieval_policy.max_context_chars or 6000)
    if max_chars <= 0:
        max_chars = 6000
    normalized = _normalize_items(evidence_items)

    deduped: dict[str, EvidenceItem] = {}
    for item in normalized:
        key = _dedup_key(item)
        prev = deduped.get(key)
        prev_score = prev.score if prev and prev.score is not None else float("-inf")
        current_score = item.score if item.score is not None else float("-inf")
        if prev is None or current_score > prev_score:
            deduped[key] = item

    ranked = sorted(
        deduped.values(),
        key=lambda x: x.score if x.score is not None else float("-inf"),
        reverse=True,
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_type_counts: dict[str, int] = defaultdict(int)

    selected: list[EvidenceItem] = []
    context_lines: list[str] = []
    char_budget = 0

    for item in ranked:
        title_prefix = f"{item.title}: " if item.title else ""
        line = f"[{item.source_type}] {title_prefix}{item.content}".strip()
        line_with_newline = line + "\n"
        if char_budget + len(line_with_newline) > max_chars:
            break
        char_budget += len(line_with_newline)
        context_lines.append(line_with_newline)
        selected.append(item)

    context_text = "".join(context_lines).strip()
    context_text, truncated = _truncate(context_text, max_chars)

    for item in selected:
        dump = item.model_dump(mode="json")
        grouped[item.source_type].append(dump)
        source_type_counts[item.source_type] += 1

    scores = [item.score for item in selected if item.score is not None]
    score_summary = {
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
        "avg": (sum(scores) / len(scores)) if scores else None,
    }
    compact_graph_paths, graph_summary, graph_context_preview = _build_graph_context(
        graph_paths,
        max_chars,
    )

    return {
        "query": query,
        "intent": profile.intent.value,
        "domain": profile.domain.value,
        "total_evidence": len(normalized),
        "selected_evidence_count": len(selected),
        "source_type_counts": dict(source_type_counts),
        "grouped_evidence": dict(grouped),
        "evidence": [item.model_dump(mode="json") for item in selected],
        "context": context_text,
        "context_preview": context_text[:300],
        "max_context_chars": max_chars,
        "truncated": truncated,
        "score_summary": score_summary,
        "graph_paths": compact_graph_paths,
        "graph_summary": graph_summary,
        "graph_context_preview": graph_context_preview,
    }
