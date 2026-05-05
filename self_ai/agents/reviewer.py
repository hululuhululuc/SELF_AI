# coding=utf-8
"""Storage-driven reviewer agent for Phase 6B."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from ..text_utils import shorten_text

_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)```|```\s*([\s\S]*?)```")


class ReviewerInput(BaseModel):
    run_id: str = ""
    session_id: str = "default"
    task: str = ""
    task_profile: dict[str, Any] = Field(default_factory=dict)
    workflow_decision: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    context_pack: dict[str, Any] = Field(default_factory=dict)
    agent_outputs: dict[str, Any] = Field(default_factory=dict)
    redis_context: dict[str, Any] = Field(default_factory=dict)
    qdrant_review_memory: list[dict[str, Any]] = Field(default_factory=list)
    neo4j_graph_context: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _normalize_level(value: Any, *, default: str = "medium") -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        return str(value.value)
    text = str(value).strip().lower()
    return text if text else default


def _as_list_str(value: Any, *, limit: int = 8, max_chars: int = 220) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:limit]:
        text = shorten_text(str(item), max_chars=max_chars).strip()
        if text:
            out.append(text)
    return out


def _compact_evidence_refs(value: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, Any]] = []
    for raw in value[:limit]:
        if not isinstance(raw, dict):
            continue
        refs.append(
            {
                "source_type": str(raw.get("source_type", "unknown")),
                "ref": raw.get("ref") or raw.get("chunk_id") or raw.get("path") or raw.get("url"),
                "score": raw.get("score"),
                "preview": shorten_text(str(raw.get("preview", "")), max_chars=140),
            }
        )
    return refs


def _compact_storage_evidence(value: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for raw in value[:limit]:
        if not isinstance(raw, dict):
            continue
        evidence.append(
            {
                "content": shorten_text(str(raw.get("content", "")), max_chars=180),
                "source_type": str(raw.get("source_type", "review")),
                "source": raw.get("source"),
                "path": raw.get("path"),
                "symbol": raw.get("symbol"),
                "chunk_id": raw.get("chunk_id"),
                "score": raw.get("score"),
                "url": raw.get("url"),
                "metadata": raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
            }
        )
    return evidence


def _extract_json(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _JSON_BLOCK_RE.search(stripped)
    if fenced:
        candidate = fenced.group(1) or fenced.group(2) or ""
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("invalid_review_json")


class ReviewerAgent:
    """Storage-driven reviewer that outputs structured ReviewReport dict."""

    name = "reviewer"
    role = "reviewer"

    def build_prompt(self, review_input: ReviewerInput) -> str:
        risk_level = _normalize_level(review_input.task_profile.get("risk_level"), default="medium")
        complexity = _normalize_level(review_input.task_profile.get("complexity"), default="medium")
        policy_suggested = review_input.workflow_decision.get("selected_agents", [])
        executed = review_input.metadata.get("executed_role_agents", [])
        return (
            "You are a storage-driven reviewer for a coding/research workflow.\n"
            "Review outputs against task, plan, context, and storage evidence.\n"
            "Do NOT modify code.\n"
            "Do NOT re-classify task intent.\n"
            "Do NOT change workflow decision.\n"
            "Return ONLY valid JSON.\n"
            "Required keys: pass_review, severity, major_issues, minor_issues, missing_requirements, "
            "requires_revision, revision_target_agent, revision_instruction, evidence_refs, storage_evidence, rationale.\n\n"
            f"Task: {shorten_text(review_input.task, max_chars=500)}\n"
            f"Risk: {risk_level}\n"
            f"Complexity: {complexity}\n"
            f"Policy suggested agents: {policy_suggested}\n"
            f"Executed role agents: {executed}\n"
            f"Plan step count: {len(review_input.plan.get('steps', [])) if isinstance(review_input.plan, dict) else 0}\n"
            f"Role output count: {len(review_input.agent_outputs) if isinstance(review_input.agent_outputs, dict) else 0}\n"
            f"Qdrant review memory count: {len(review_input.qdrant_review_memory)}\n"
            f"Neo4j graph context count: {len(review_input.neo4j_graph_context)}\n"
        )

    def _fallback_report(
        self,
        review_input: ReviewerInput,
        *,
        error_type: str,
    ) -> dict[str, Any]:
        risk_level = _normalize_level(review_input.task_profile.get("risk_level"), default="medium")
        complexity = _normalize_level(review_input.task_profile.get("complexity"), default="medium")
        is_high = risk_level == "high" or complexity == "high"

        pass_review = not is_high
        severity = "high" if is_high else "low"
        major_issues = (
            ["reviewer_unavailable_for_high_risk_task"]
            if is_high
            else []
        )
        minor_issues = (
            []
            if is_high
            else ["reviewer_fallback_used_low_risk_task"]
        )
        requires_revision = is_high
        revision_target = "synthesizer" if is_high else None
        revision_instruction = (
            "Reviewer fallback on high-risk task: please re-check outputs against plan and evidence."
            if is_high
            else ""
        )
        return {
            "run_id": review_input.run_id,
            "session_id": review_input.session_id,
            "reviewer_name": self.name,
            "review_type": "storage_driven",
            "pass_review": pass_review,
            "severity": severity,
            "major_issues": major_issues,
            "minor_issues": minor_issues,
            "missing_requirements": [],
            "requires_revision": requires_revision,
            "revision_target_agent": revision_target,
            "revision_instruction": revision_instruction,
            "evidence_refs": [],
            "storage_evidence": [],
            "rationale": "fallback_review_due_to_model_or_parse_error",
            "risk_level": risk_level,
            "metadata": {
                "fallback": True,
                "error_type": error_type,
                "policy_suggested_agents": review_input.workflow_decision.get("selected_agents", []),
                "executed_role_agents": review_input.metadata.get("executed_role_agents", []),
            },
        }

    def _normalize_report(self, review_input: ReviewerInput, raw: dict[str, Any]) -> dict[str, Any]:
        risk_level = _normalize_level(review_input.task_profile.get("risk_level"), default="medium")
        severity = _normalize_level(raw.get("severity"), default="medium")
        if severity not in {"low", "medium", "high", "critical"}:
            severity = "medium"

        major_issues = _as_list_str(raw.get("major_issues"), limit=10)
        minor_issues = _as_list_str(raw.get("minor_issues"), limit=10)
        missing_reqs = _as_list_str(raw.get("missing_requirements"), limit=10)
        requires_revision = bool(raw.get("requires_revision", False))
        pass_review = bool(raw.get("pass_review", False))
        if major_issues or missing_reqs:
            pass_review = False

        revision_target = raw.get("revision_target_agent")
        if revision_target is not None:
            revision_target = str(revision_target).strip() or None

        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "run_id": review_input.run_id,
            "session_id": review_input.session_id,
            "reviewer_name": self.name,
            "review_type": "storage_driven",
            "pass_review": pass_review,
            "severity": severity,
            "major_issues": major_issues,
            "minor_issues": minor_issues,
            "missing_requirements": missing_reqs,
            "requires_revision": requires_revision,
            "revision_target_agent": revision_target,
            "revision_instruction": shorten_text(str(raw.get("revision_instruction", "")), max_chars=500),
            "evidence_refs": _compact_evidence_refs(raw.get("evidence_refs")),
            "storage_evidence": _compact_storage_evidence(raw.get("storage_evidence")),
            "rationale": shorten_text(str(raw.get("rationale", "")), max_chars=500),
            "risk_level": risk_level,
            "metadata": {
                **metadata,
                "policy_suggested_agents": review_input.workflow_decision.get("selected_agents", []),
                "executed_role_agents": review_input.metadata.get("executed_role_agents", []),
            },
        }

    async def run(
        self,
        review_input: ReviewerInput,
        *,
        route_model_func: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        if route_model_func is None:
            from ..router import route_model as route_model_func  # lazy import

        prompt = self.build_prompt(review_input)
        try:
            model_result = await route_model_func(prompt, "reasoning")
            parsed = _extract_json(str(model_result.get("response", "")))
            report = self._normalize_report(review_input, parsed)
            report["metadata"]["model"] = model_result.get("model", "")
            return report
        except Exception as exc:
            return self._fallback_report(review_input, error_type=type(exc).__name__)

