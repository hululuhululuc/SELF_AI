# coding=utf-8
"""Task graph builders for Neo4j structural memory."""

from __future__ import annotations

import hashlib
from typing import Any

from .graph_models import GraphEdge, GraphNode, TaskGraphRecord


def _preview(value: str, max_chars: int = 200) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _extract_evidence_refs(state: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in _safe_list(state.get("research_results", [])):
        if not isinstance(item, dict):
            continue
        raw = str(item.get("content") or "")
        preview = _preview(raw)
        digest_source = raw or str(item.get("chunk_id") or item.get("id") or preview)
        digest = _hash_text(digest_source)
        refs.append(
            {
                "id": f"evidence:{digest}",
                "source_type": str(item.get("source_type") or item.get("source") or "unknown"),
                "source": item.get("source"),
                "path": item.get("path"),
                "url": item.get("url"),
                "chunk_id": item.get("chunk_id") or item.get("id"),
                "score": item.get("score"),
                "preview": preview,
                "hash": digest,
            }
        )
    return refs


def _extract_agent_outputs(state: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    outputs = state.get("agent_outputs", {})
    records: list[dict[str, Any]] = []

    if isinstance(outputs, dict):
        iterable = outputs.items()
    else:
        iterable = []

    for agent_name, value in iterable:
        if isinstance(value, dict):
            text = str(value.get("response") or value.get("content") or value.get("summary") or "")
        else:
            text = str(value)
        preview = _preview(text)
        digest = _hash_text(preview or f"{run_id}:{agent_name}")
        records.append(
            {
                "id": f"agent_output:{run_id}:{agent_name}",
                "agent_name": str(agent_name),
                "preview": preview,
                "hash": digest,
            }
        )
    return records


def _extract_review_reports(state: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, report in enumerate(_safe_list(state.get("review_reports", []))):
        if not isinstance(report, dict):
            continue
        revision = str(report.get("revision_instruction") or "")
        preview = _preview(revision)
        digest = _hash_text(preview or f"{run_id}:review:{idx}")
        records.append(
            {
                "id": str(report.get("report_id") or f"review:{run_id}:{idx}"),
                "pass_review": bool(report.get("pass_review", False)),
                "major_issue_count": len(_safe_list(report.get("major_issues", []))),
                "minor_issue_count": len(_safe_list(report.get("minor_issues", []))),
                "missing_requirement_count": len(
                    _safe_list(report.get("missing_requirements", []))
                ),
                "risk_level": str(report.get("risk_level") or "unknown"),
                "preview": preview,
                "hash": digest,
            }
        )
    return records


def build_task_graph_record(state: dict[str, Any]) -> TaskGraphRecord:
    """Build a compact task graph record from runtime state."""
    run_id = str(state.get("run_id") or "unknown-run")
    session_id = str(state.get("session_id") or "default")
    profile = state.get("task_profile", {})
    if not isinstance(profile, dict):
        profile = {}

    task_preview = _preview(str(state.get("task") or state.get("input") or ""))

    return TaskGraphRecord(
        run_id=run_id,
        session_id=session_id,
        task_id=f"task:{run_id}",
        task_profile_id=f"task_profile:{run_id}",
        task_preview=task_preview,
        intent=str(profile.get("intent") or ""),
        domain=str(profile.get("domain") or ""),
        evidence_refs=_extract_evidence_refs(state),
        agent_outputs=_extract_agent_outputs(state, run_id),
        review_reports=_extract_review_reports(state, run_id),
        metadata={"task_hash": _hash_text(task_preview or run_id)},
    )


def task_graph_to_nodes_edges(
    record: TaskGraphRecord,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Convert TaskGraphRecord into graph nodes and edges."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    task_node = GraphNode(
        id=record.task_id,
        label="Task",
        properties={
            "run_id": record.run_id,
            "session_id": record.session_id,
            "preview": record.task_preview,
            "intent": record.intent,
            "domain": record.domain,
            "task_hash": record.metadata.get("task_hash"),
            "created_at": record.created_at,
        },
    )
    nodes.append(task_node)

    profile_node = GraphNode(
        id=record.task_profile_id,
        label="TaskProfile",
        properties={
            "run_id": record.run_id,
            "session_id": record.session_id,
            "intent": record.intent,
            "domain": record.domain,
            "created_at": record.created_at,
        },
    )
    nodes.append(profile_node)
    edges.append(
        GraphEdge(
            source_id=record.task_id,
            target_id=record.task_profile_id,
            relation="HAS_PROFILE",
            properties={"run_id": record.run_id},
        )
    )

    for evidence in record.evidence_refs:
        evidence_id = str(evidence.get("id") or "")
        if not evidence_id:
            continue
        nodes.append(
            GraphNode(
                id=evidence_id,
                label="Evidence",
                properties={
                    "run_id": record.run_id,
                    "source_type": evidence.get("source_type"),
                    "source": evidence.get("source"),
                    "path": evidence.get("path"),
                    "url": evidence.get("url"),
                    "chunk_id": evidence.get("chunk_id"),
                    "score": evidence.get("score"),
                    "preview": evidence.get("preview"),
                    "content_hash": evidence.get("hash"),
                },
            )
        )
        edges.append(
            GraphEdge(
                source_id=record.task_id,
                target_id=evidence_id,
                relation="USED_EVIDENCE",
                properties={"run_id": record.run_id},
            )
        )

    for output in record.agent_outputs:
        output_id = str(output.get("id") or "")
        if not output_id:
            continue
        nodes.append(
            GraphNode(
                id=output_id,
                label="AgentRun",
                properties={
                    "run_id": record.run_id,
                    "agent_name": output.get("agent_name"),
                    "preview": output.get("preview"),
                    "output_hash": output.get("hash"),
                },
            )
        )
        edges.append(
            GraphEdge(
                source_id=record.task_id,
                target_id=output_id,
                relation="HAS_OUTPUT",
                properties={"run_id": record.run_id},
            )
        )

    for report in record.review_reports:
        report_id = str(report.get("id") or "")
        if not report_id:
            continue
        nodes.append(
            GraphNode(
                id=report_id,
                label="ReviewReport",
                properties={
                    "run_id": record.run_id,
                    "pass_review": report.get("pass_review"),
                    "major_issue_count": report.get("major_issue_count"),
                    "minor_issue_count": report.get("minor_issue_count"),
                    "missing_requirement_count": report.get("missing_requirement_count"),
                    "risk_level": report.get("risk_level"),
                    "preview": report.get("preview"),
                    "report_hash": report.get("hash"),
                },
            )
        )
        edges.append(
            GraphEdge(
                source_id=record.task_id,
                target_id=report_id,
                relation="HAS_REVIEW",
                properties={"run_id": record.run_id},
            )
        )

    return nodes, edges
