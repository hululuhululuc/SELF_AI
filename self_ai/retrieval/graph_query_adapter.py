# coding=utf-8
"""Read-only graph query adapter for Phase 4.5."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..graph.graph_models import GraphPath
from ..schemas import EvidenceItem, RetrievalPolicy, TaskIntent, TaskProfile
from ..text_utils import normalize_text

_HISTORY_HINT_KEYWORDS = (
    "history",
    "historical",
    "past decision",
    "prior decision",
    "project structure",
    "历史结论",
    "历史决策",
    "项目结构",
    "之前怎么做",
    "过往方案",
)

_ISSUE_HINT_KEYWORDS = (
    "error",
    "exception",
    "bug",
    "failure",
    "traceback",
    "报错",
    "异常",
    "故障",
    "修复",
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

_QUERY_TYPE_RELATED_SYMBOLS = "find_related_symbols"
_QUERY_TYPE_PRIOR_ISSUES = "find_prior_issues"
_QUERY_TYPE_TASK_LINEAGE = "find_task_lineage"


class GraphQueryPlan(BaseModel):
    enabled: bool = False
    query_types: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    issue_text: str | None = None
    run_id: str | None = None
    max_paths: int = 5
    rationale: str = ""


def _normalize_evidence_items(
    items: list[EvidenceItem | dict[str, Any]],
) -> list[EvidenceItem]:
    normalized: list[EvidenceItem] = []
    for item in items:
        if isinstance(item, EvidenceItem):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            try:
                normalized.append(EvidenceItem(**item))
            except Exception:
                continue
    return normalized


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _extract_query_symbols(query: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(query or "")
    symbols: list[str] = []
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "please",
        "write",
        "create",
        "python",
        "java",
        "javascript",
        "function",
        "class",
        "file",
        "main",
        "error",
        "issue",
    }
    for token in tokens:
        t = token.strip("_")
        if len(t) < 3:
            continue
        if t.lower() in stop_words:
            continue
        # Treat snake_case/camelCase/mixed names as likely symbols.
        if "_" in t or any(ch.isupper() for ch in t[1:]):
            symbols.append(t)
    return _unique_keep_order(symbols)


def _extract_symbols(
    query: str, evidence_items: list[EvidenceItem | dict[str, Any]]
) -> list[str]:
    evidence_symbols: list[str] = []
    for item in _normalize_evidence_items(evidence_items):
        symbol = (item.symbol or "").strip()
        if symbol:
            evidence_symbols.append(symbol)
    return _unique_keep_order([*evidence_symbols, *_extract_query_symbols(query)])


def _contains_history_hint(query: str) -> bool:
    lower = (query or "").lower()
    return any(keyword in lower for keyword in _HISTORY_HINT_KEYWORDS)


def _contains_issue_hint(query: str) -> bool:
    lower = (query or "").lower()
    return any(keyword in lower for keyword in _ISSUE_HINT_KEYWORDS)


def _task_intent_value(profile: TaskProfile) -> str:
    return profile.intent.value if hasattr(profile.intent, "value") else str(profile.intent)


def _task_run_id(profile: TaskProfile) -> str | None:
    identity = getattr(profile, "identity", None)
    if identity is None:
        return None
    run_id = getattr(identity, "run_id", None)
    return str(run_id) if run_id else None


def _path_signature(path: GraphPath) -> str:
    node_key = "|".join(node.id for node in path.nodes)
    edge_key = "|".join(f"{edge.source_id}:{edge.relation}:{edge.target_id}" for edge in path.edges)
    summary = str(path.metadata.get("summary", "")) if isinstance(path.metadata, dict) else ""
    return f"{node_key}::{edge_key}::{summary[:80]}"


def _normalize_path(path: GraphPath | dict[str, Any]) -> GraphPath | None:
    if isinstance(path, GraphPath):
        return path
    if isinstance(path, dict):
        try:
            return GraphPath(**path)
        except Exception:
            return None
    return None


def _path_score(path: GraphPath) -> float:
    metadata = path.metadata if isinstance(path.metadata, dict) else {}
    raw_score = metadata.get("score")
    try:
        score = float(raw_score) if raw_score is not None else 0.0
    except Exception:
        score = 0.0
    return score


def _path_relation_priority(path: GraphPath) -> int:
    priority = {
        "CALLS": 5,
        "FIXED_BY": 4,
        "TESTS": 3,
        "HAS_REVIEW": 2,
        "USED_EVIDENCE": 1,
    }
    result = 0
    for edge in path.edges:
        result = max(result, priority.get(edge.relation, 0))
    return result


class GraphQueryAdapter:
    """Intent-aware read-only graph query adapter."""

    def __init__(
        self,
        graph_memory: Any,
        *,
        max_paths: int = 5,
        enabled: bool = True,
    ) -> None:
        self.graph_memory = graph_memory
        self.max_paths = max(1, int(max_paths))
        self.enabled = enabled

    def plan_graph_queries(
        self,
        *,
        query: str,
        task_profile: TaskProfile,
        retrieval_policy: RetrievalPolicy,
        evidence_items: list[EvidenceItem | dict[str, Any]],
    ) -> GraphQueryPlan:
        normalized_query = normalize_text(query)
        if not self.enabled:
            return GraphQueryPlan(enabled=False, rationale="graph adapter disabled")
        if not retrieval_policy.use_graph:
            return GraphQueryPlan(enabled=False, rationale="retrieval_policy.use_graph is false")

        intent = _task_intent_value(task_profile)
        run_id = _task_run_id(task_profile)
        symbols = _extract_symbols(normalized_query, evidence_items)
        query_types: list[str] = []
        issue_text: str | None = None

        if intent in {
            TaskIntent.DEBUGGING.value,
            TaskIntent.CODE_REVIEW.value,
            TaskIntent.CODE_MODIFICATION.value,
        }:
            query_types = [
                _QUERY_TYPE_RELATED_SYMBOLS,
                _QUERY_TYPE_PRIOR_ISSUES,
                _QUERY_TYPE_TASK_LINEAGE,
            ]
            issue_text = normalized_query
        elif intent in {TaskIntent.ARCHITECTURE_DESIGN.value, TaskIntent.PLANNING.value}:
            query_types = [_QUERY_TYPE_TASK_LINEAGE, _QUERY_TYPE_RELATED_SYMBOLS]
        elif intent == TaskIntent.CODE_GENERATION.value:
            if symbols:
                query_types.append(_QUERY_TYPE_RELATED_SYMBOLS)
            if _contains_issue_hint(normalized_query):
                query_types.append(_QUERY_TYPE_PRIOR_ISSUES)
                issue_text = normalized_query
        elif intent in {TaskIntent.RESEARCH_SUMMARY.value, TaskIntent.RAG_ANSWERING.value}:
            query_types = [_QUERY_TYPE_TASK_LINEAGE]
        elif intent in {TaskIntent.DOCUMENT_WRITING.value, TaskIntent.GENERAL_QA.value}:
            if _contains_history_hint(normalized_query):
                query_types = [_QUERY_TYPE_TASK_LINEAGE]
            else:
                return GraphQueryPlan(
                    enabled=False,
                    rationale="document/general intent without explicit history hint",
                )

        query_types = _unique_keep_order(query_types)
        if not query_types:
            return GraphQueryPlan(
                enabled=False,
                symbols=symbols,
                run_id=run_id,
                rationale="no graph query type for intent",
            )

        max_paths = max(1, min(self.max_paths, int(retrieval_policy.top_k_graph or self.max_paths)))
        return GraphQueryPlan(
            enabled=True,
            query_types=query_types,
            symbols=symbols,
            issue_text=issue_text,
            run_id=run_id,
            max_paths=max_paths,
            rationale=f"intent={intent}",
        )

    def run_graph_queries(self, plan: GraphQueryPlan) -> list[GraphPath]:
        if not self.enabled or not plan.enabled or not plan.query_types:
            return []

        gathered: list[GraphPath] = []
        for query_type in plan.query_types:
            if len(gathered) >= plan.max_paths:
                break
            try:
                if query_type == _QUERY_TYPE_RELATED_SYMBOLS:
                    for symbol in plan.symbols:
                        paths = self.graph_memory.find_related_symbols(symbol, limit=plan.max_paths)
                        for raw_path in paths or []:
                            normalized = _normalize_path(raw_path)
                            if normalized is not None:
                                gathered.append(normalized)
                            if len(gathered) >= plan.max_paths:
                                break
                        if len(gathered) >= plan.max_paths:
                            break
                elif query_type == _QUERY_TYPE_PRIOR_ISSUES:
                    if plan.issue_text:
                        paths = self.graph_memory.find_prior_issues(plan.issue_text, limit=plan.max_paths)
                        for raw_path in paths or []:
                            normalized = _normalize_path(raw_path)
                            if normalized is not None:
                                gathered.append(normalized)
                            if len(gathered) >= plan.max_paths:
                                break
                elif query_type == _QUERY_TYPE_TASK_LINEAGE:
                    if plan.run_id:
                        path = self.graph_memory.find_task_lineage(plan.run_id, limit=plan.max_paths)
                        normalized = _normalize_path(path)
                        if normalized is not None:
                            gathered.append(normalized)
            except Exception:
                continue

        deduped: dict[str, GraphPath] = {}
        for path in gathered:
            deduped[_path_signature(path)] = path

        ranked = sorted(
            deduped.values(),
            key=lambda item: (
                _path_score(item),
                _path_relation_priority(item),
                -len(item.edges),
            ),
            reverse=True,
        )
        return ranked[: plan.max_paths]
