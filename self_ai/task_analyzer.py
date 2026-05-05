# coding=utf-8
"""Structured task analyzer for Phase 1 storage-aware workflow."""

import json
import re
from typing import Any

from .observability import trace
from .router import route_model
from .schemas import (
    ComplexityLevel,
    ExpectedArtifact,
    RiskLevel,
    RunIdentity,
    SurfaceSignals,
    TaskDomain,
    TaskIntent,
    TaskProfile,
)
from .storage_policy import enrich_profile_with_storage_policies
from .text_utils import extract_surface_signals, normalize_text


_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)```|```\s*([\s\S]*?)```")


def _build_classifier_prompt(input_text: str) -> str:
    return (
        "You are a task classifier for a storage-aware multi-agent coding and "
        "research workflow.\n\n"
        "Classify the user's task by intent and workflow needs, not by keywords.\n\n"
        "Return only valid JSON with these fields:\n"
        "intent, domain, requires_code, requires_repo_context, requires_research,\n"
        "requires_planning, requires_review, expected_artifact, complexity,\n"
        "risk_level, confidence, rationale.\n\n"
        "Allowed intent:\n"
        "code_generation, code_modification, debugging, code_review,\n"
        "architecture_design, research_summary, rag_answering,\n"
        "document_writing, planning, general_qa\n\n"
        "Allowed domain:\n"
        "software, ai_agent, rag, data, academic, general\n\n"
        "Allowed expected_artifact:\n"
        "answer, code_patch, code_review_report, design_doc,\n"
        "implementation_plan, research_report, test_plan\n\n"
        "Rules:\n"
        "- Project workflow redesign, multi-agent orchestration, storage design, "
        "or architecture plans should be architecture_design or planning.\n"
        "- Inspecting existing code for defects should be code_review or debugging.\n"
        "- Fixing runtime errors should be debugging.\n"
        "- Producing concrete code changes should be code_modification.\n"
        "- Producing a standalone function/script should be code_generation.\n"
        "- Summarizing sources, papers, journals, or current information should be "
        "research_summary.\n"
        "- Academic rewriting or translation should be document_writing.\n"
        "- If uncertain, set confidence < 0.7, requires_planning=true, "
        "requires_review=true.\n\n"
        "Do not include markdown.\n"
        "Do not include explanation outside JSON.\n\n"
        f"User input:\n{input_text}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
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
        candidate = stripped[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Unable to extract valid JSON object from model response.")


def _profile_trace_fields(profile: TaskProfile) -> dict[str, Any]:
    return {
        "run_id": profile.identity.run_id,
        "session_id": profile.identity.session_id,
        "intent": profile.intent.value,
        "domain": profile.domain.value,
        "complexity": profile.complexity.value,
        "risk_level": profile.risk_level.value,
        "confidence": profile.confidence,
        "requires_code": profile.requires_code,
        "requires_planning": profile.requires_planning,
        "requires_review": profile.requires_review,
        "write_redis": profile.memory_write_policy.write_redis,
        "write_qdrant": profile.memory_write_policy.write_qdrant,
        "write_neo4j": profile.memory_write_policy.write_neo4j,
    }


def validate_task_profile(profile: TaskProfile) -> TaskProfile:
    """Normalize and enforce internal consistency for profile fields."""
    intent = profile.intent

    if intent == TaskIntent.CODE_GENERATION:
        profile.requires_code = True
        if profile.expected_artifact == ExpectedArtifact.ANSWER:
            profile.expected_artifact = ExpectedArtifact.CODE_PATCH

    if intent == TaskIntent.CODE_MODIFICATION:
        profile.requires_code = True
        profile.requires_repo_context = True
        profile.requires_review = True
        profile.expected_artifact = ExpectedArtifact.CODE_PATCH

    if intent == TaskIntent.DEBUGGING:
        profile.requires_code = True
        profile.requires_repo_context = True
        profile.requires_review = True
        if profile.expected_artifact not in {
            ExpectedArtifact.CODE_REVIEW_REPORT,
            ExpectedArtifact.CODE_PATCH,
        }:
            profile.expected_artifact = ExpectedArtifact.CODE_REVIEW_REPORT

    if intent == TaskIntent.CODE_REVIEW:
        profile.requires_code = True
        profile.requires_repo_context = True
        profile.requires_review = True
        profile.expected_artifact = ExpectedArtifact.CODE_REVIEW_REPORT

    if intent == TaskIntent.ARCHITECTURE_DESIGN:
        profile.requires_planning = True
        profile.requires_review = True
        if profile.complexity == ComplexityLevel.LOW:
            profile.complexity = ComplexityLevel.MEDIUM
        if profile.expected_artifact not in {
            ExpectedArtifact.DESIGN_DOC,
            ExpectedArtifact.IMPLEMENTATION_PLAN,
        }:
            profile.expected_artifact = ExpectedArtifact.DESIGN_DOC

    if intent == TaskIntent.PLANNING:
        profile.requires_planning = True
        profile.requires_review = True
        profile.expected_artifact = ExpectedArtifact.IMPLEMENTATION_PLAN

    if intent == TaskIntent.RESEARCH_SUMMARY:
        profile.requires_research = True
        profile.expected_artifact = ExpectedArtifact.RESEARCH_REPORT

    if intent == TaskIntent.RAG_ANSWERING:
        profile.requires_research = True

    if profile.complexity == ComplexityLevel.HIGH:
        profile.requires_planning = True
        profile.requires_review = True

    if profile.confidence < 0:
        profile.confidence = 0.0
    if profile.confidence > 1:
        profile.confidence = 1.0

    return profile


def fallback_analyze_task(
    input_text: str,
    *,
    session_id: str = "default",
    user_id: str | None = None,
) -> TaskProfile:
    """Fallback analyzer when model classification fails."""
    normalized = normalize_text(input_text)
    signals: SurfaceSignals = extract_surface_signals(normalized)
    lower = normalized.lower()
    identity = RunIdentity(session_id=session_id, user_id=user_id)

    intent = TaskIntent.GENERAL_QA
    domain = TaskDomain.GENERAL
    expected_artifact = ExpectedArtifact.ANSWER
    complexity = ComplexityLevel.MEDIUM
    risk_level = RiskLevel.MEDIUM
    requires_code = False
    requires_repo_context = False
    requires_research = False
    requires_planning = False
    requires_review = False
    confidence = 0.55
    rationale = "Fallback heuristic classification."

    if any(k in lower for k in ["重构", "架构", "workflow", "编排", "多智能体", "agent"]):
        intent = TaskIntent.ARCHITECTURE_DESIGN
        domain = TaskDomain.AI_AGENT if ("智能体" in normalized or "agent" in lower) else TaskDomain.SOFTWARE
        expected_artifact = ExpectedArtifact.DESIGN_DOC
        complexity = ComplexityLevel.HIGH
        requires_planning = True
        requires_review = True
        confidence = 0.62
        rationale = "Detected architecture/workflow redesign intent."
    elif any(k in lower for k in ["报错", "error", "exception", "bug", "valueerror", "traceback", "定位"]):
        intent = TaskIntent.DEBUGGING
        domain = TaskDomain.SOFTWARE
        expected_artifact = ExpectedArtifact.CODE_REVIEW_REPORT
        requires_code = True
        requires_repo_context = True
        requires_review = True
        confidence = 0.72
        rationale = "Detected debugging/error diagnosis intent."
    elif any(k in lower for k in ["写一个", "函数", "python", "脚本", "实现"]):
        intent = TaskIntent.CODE_GENERATION
        domain = TaskDomain.SOFTWARE
        expected_artifact = ExpectedArtifact.CODE_PATCH
        complexity = ComplexityLevel.LOW
        requires_code = True
        confidence = 0.74
        rationale = "Detected standalone code generation intent."
    elif any(k in lower for k in ["总结", "投稿要求", "paper", "journal", "文献", "调研"]):
        intent = TaskIntent.RESEARCH_SUMMARY
        domain = TaskDomain.ACADEMIC if any(
            k in lower for k in ["journal", "paper", "投稿", "期刊", "学术"]
        ) else TaskDomain.GENERAL
        expected_artifact = ExpectedArtifact.RESEARCH_REPORT
        requires_research = True
        confidence = 0.7
        rationale = "Detected research summary intent."
    elif any(k in lower for k in ["翻译", "润色", "摘要", "英文", "academic", "期刊"]):
        intent = TaskIntent.DOCUMENT_WRITING
        domain = TaskDomain.ACADEMIC
        expected_artifact = ExpectedArtifact.ANSWER
        confidence = 0.69
        rationale = "Detected academic writing/translation intent."

    profile = TaskProfile(
        identity=identity,
        intent=intent,
        domain=domain,
        requires_code=requires_code,
        requires_repo_context=requires_repo_context,
        requires_research=requires_research,
        requires_planning=requires_planning,
        requires_review=requires_review,
        expected_artifact=expected_artifact,
        complexity=complexity,
        risk_level=risk_level,
        confidence=confidence,
        rationale=rationale,
        surface_signals=signals,
    )
    profile = validate_task_profile(profile)
    return enrich_profile_with_storage_policies(profile)


async def analyze_task(
    input_text: str,
    *,
    session_id: str = "default",
    user_id: str | None = None,
) -> TaskProfile:
    """Analyze input task into a structured TaskProfile."""
    normalized = normalize_text(input_text)
    preview = normalized[:120]
    trace("task_analyzer.start", input_preview=preview, session_id=session_id)

    signals = extract_surface_signals(normalized)
    trace(
        "task_analyzer.surface_signals",
        session_id=session_id,
        has_code_block=signals.has_code_block,
        has_file_path=signals.has_file_path,
        has_error_trace=signals.has_error_trace,
        has_url=signals.has_url,
        has_chinese=signals.has_chinese,
        has_markdown_table=signals.has_markdown_table,
        length=signals.length,
    )

    try:
        prompt = _build_classifier_prompt(normalized)
        model_result = await route_model(prompt, "fast")
        model_response = str(model_result.get("response", ""))
        trace(
            "task_analyzer.model_response",
            session_id=session_id,
            model=model_result.get("model", ""),
            response_preview=model_response[:200],
        )

        parsed = _extract_json_object(model_response)
        profile = TaskProfile(
            identity=RunIdentity(session_id=session_id, user_id=user_id),
            surface_signals=signals,
            **parsed,
        )
        profile = validate_task_profile(profile)
        profile = enrich_profile_with_storage_policies(profile)
    except Exception as exc:
        trace(
            "task_analyzer.parse_error",
            session_id=session_id,
            error=type(exc).__name__,
            message=str(exc)[:200],
        )
        profile = fallback_analyze_task(
            normalized,
            session_id=session_id,
            user_id=user_id,
        )
        trace(
            "task_analyzer.fallback",
            run_id=profile.identity.run_id,
            session_id=profile.identity.session_id,
            input_preview=preview,
        )

    trace("task_analyzer.storage_policy", **_profile_trace_fields(profile))
    trace("task_analyzer.end", **_profile_trace_fields(profile))
    return profile
