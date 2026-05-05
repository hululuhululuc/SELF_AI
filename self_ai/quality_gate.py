# coding=utf-8
"""Quality gate decision logic for Phase 6B."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QualityGateDecision(BaseModel):
    run_id: str = ""
    session_id: str = "default"
    passed: bool = False
    decision: str = "revise"  # pass | warn | revise | fail
    severity: str = "medium"
    requires_revision: bool = False
    revision_target_agent: str | None = None
    revision_instruction: str = ""
    max_revision_iterations: int = 1
    current_iteration: int = 0
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rationale: str = ""
    policy_version: str = "phase6b.v1"
    metadata: dict[str, Any] = Field(default_factory=dict)


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v).strip()]


def _normalize_severity(value: Any, *, default: str = "medium") -> str:
    text = str(value).strip().lower() if value is not None else default
    if text not in {"low", "medium", "high", "critical"}:
        return default
    return text


def _risk_level(task_profile: dict[str, Any] | None) -> str:
    if not isinstance(task_profile, dict):
        return "medium"
    value = task_profile.get("risk_level", "medium")
    if hasattr(value, "value"):
        value = value.value
    return _normalize_severity(value, default="medium")


def decide_quality_gate(
    *,
    review_report: dict[str, Any] | None,
    task_profile: dict[str, Any] | None = None,
    run_id: str = "",
    session_id: str = "default",
    max_revision_iterations: int = 1,
    current_iteration: int = 0,
) -> QualityGateDecision:
    """Decide pass/warn/revise/fail from ReviewReport and risk rules."""
    report = review_report if isinstance(review_report, dict) else {}
    report_severity = _normalize_severity(report.get("severity"), default="medium")
    major = _as_list(report.get("major_issues"))
    minor = _as_list(report.get("minor_issues"))
    missing = _as_list(report.get("missing_requirements"))
    requires_revision = bool(report.get("requires_revision", False))
    pass_review = bool(report.get("pass_review", False))
    revision_target = report.get("revision_target_agent")
    revision_instruction = str(report.get("revision_instruction", "") or "")
    metadata = report.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    reviewer_failed = bool(metadata.get("error_type"))
    risk = _risk_level(task_profile)

    blocking: list[str] = []
    warnings: list[str] = []
    decision = "revise"
    passed = False
    rationale = "gate_default_revise"

    if reviewer_failed:
        warnings.append(f"reviewer_failure:{metadata.get('error_type')}")
        if risk in {"high", "critical"}:
            decision = "fail" if report_severity == "critical" else "revise"
            blocking.append("reviewer_failure_high_risk")
            passed = False
            rationale = "reviewer_failed_high_risk"
        else:
            decision = "warn" if pass_review else "pass"
            passed = True
            rationale = "reviewer_failed_low_risk_non_blocking"
    elif pass_review and not major and not missing and report_severity in {"low", "medium"} and not requires_revision:
        if minor:
            decision = "warn"
            warnings.extend(minor)
            passed = True
            rationale = "minor_issues_non_blocking"
        else:
            decision = "pass"
            passed = True
            rationale = "review_passed_clean"
    elif major or missing or requires_revision:
        decision = "revise"
        passed = False
        blocking.extend(major)
        blocking.extend(missing)
        rationale = "major_or_missing_or_requires_revision"
    elif report_severity in {"high", "critical"}:
        decision = "fail" if report_severity == "critical" else "revise"
        passed = False
        blocking.append("high_severity_review")
        rationale = "severity_blocking"
    else:
        decision = "warn"
        passed = True
        rationale = "fallback_warn"

    gate_requires_revision = decision in {"revise", "fail"}
    return QualityGateDecision(
        run_id=run_id,
        session_id=session_id,
        passed=passed,
        decision=decision,
        severity=report_severity,
        requires_revision=gate_requires_revision or requires_revision,
        revision_target_agent=(str(revision_target) if revision_target else None),
        revision_instruction=revision_instruction,
        max_revision_iterations=max(1, int(max_revision_iterations)),
        current_iteration=max(0, int(current_iteration)),
        blocking_issues=blocking,
        warnings=warnings,
        rationale=rationale,
        metadata={
            "reviewer_failed": reviewer_failed,
            "risk_level": risk,
            "policy_suggested_agents": metadata.get("policy_suggested_agents", []),
            "executed_role_agents": metadata.get("executed_role_agents", []),
        },
    )

