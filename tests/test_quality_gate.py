# coding=utf-8
"""Tests for Phase 6B quality gate decisions."""

from pathlib import Path

from self_ai.quality_gate import QualityGateDecision, decide_quality_gate


def _report(**kwargs):
    base = {
        "pass_review": True,
        "severity": "low",
        "major_issues": [],
        "minor_issues": [],
        "missing_requirements": [],
        "requires_revision": False,
        "revision_target_agent": None,
        "revision_instruction": "",
        "metadata": {},
    }
    base.update(kwargs)
    return base


def test_pass_review_no_issues_returns_pass() -> None:
    decision = decide_quality_gate(
        review_report=_report(),
        task_profile={"risk_level": "low"},
        run_id="r1",
        session_id="s1",
    )
    assert decision.decision == "pass"
    assert decision.passed is True


def test_minor_issues_returns_warn() -> None:
    decision = decide_quality_gate(
        review_report=_report(minor_issues=["style"]),
        task_profile={"risk_level": "low"},
    )
    assert decision.decision == "warn"
    assert decision.passed is True


def test_major_issues_returns_revise() -> None:
    decision = decide_quality_gate(
        review_report=_report(pass_review=False, major_issues=["bug found"]),
        task_profile={"risk_level": "medium"},
    )
    assert decision.decision == "revise"
    assert decision.passed is False


def test_missing_requirements_returns_revise() -> None:
    decision = decide_quality_gate(
        review_report=_report(pass_review=False, missing_requirements=["missing benchmark"]),
        task_profile={"risk_level": "medium"},
    )
    assert decision.decision == "revise"


def test_high_critical_severity_returns_revise_or_fail() -> None:
    high = decide_quality_gate(
        review_report=_report(pass_review=False, severity="high"),
        task_profile={"risk_level": "high"},
    )
    critical = decide_quality_gate(
        review_report=_report(pass_review=False, severity="critical"),
        task_profile={"risk_level": "high"},
    )
    assert high.decision in {"revise", "fail"}
    assert critical.decision in {"revise", "fail"}


def test_requires_revision_true_returns_revise() -> None:
    decision = decide_quality_gate(
        review_report=_report(pass_review=False, requires_revision=True, revision_target_agent="coder"),
        task_profile={"risk_level": "medium"},
    )
    assert decision.decision == "revise"
    assert decision.requires_revision is True


def test_reviewer_failure_high_risk_returns_revise_or_fail() -> None:
    decision = decide_quality_gate(
        review_report=_report(
            pass_review=False,
            severity="high",
            metadata={"error_type": "RuntimeError"},
        ),
        task_profile={"risk_level": "high"},
    )
    assert decision.decision in {"revise", "fail"}
    assert decision.passed is False


def test_reviewer_failure_low_risk_returns_warn_or_pass() -> None:
    decision = decide_quality_gate(
        review_report=_report(
            pass_review=True,
            severity="low",
            metadata={"error_type": "RuntimeError"},
        ),
        task_profile={"risk_level": "low"},
    )
    assert decision.decision in {"warn", "pass"}
    assert decision.passed is True


def test_quality_gate_source_does_not_call_model_or_storage() -> None:
    source = Path("quality_gate.py").read_text(encoding="utf-8").lower()
    assert "route_model" not in source
    assert "qdrant" not in source
    assert "neo4j" not in source


def test_quality_gate_decision_model_dump_json_stable() -> None:
    decision = QualityGateDecision(
        run_id="r1",
        session_id="s1",
        passed=True,
        decision="pass",
        severity="low",
    )
    dumped = decision.model_dump(mode="json")
    assert dumped["run_id"] == "r1"
    assert dumped["session_id"] == "s1"
    assert dumped["decision"] == "pass"

