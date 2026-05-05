# coding=utf-8
"""Tests for frontend smoke runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from self_ai.frontend_smoke import run_frontend_smoke_cases


def test_fake_smoke_runs_default_five_cases_and_writes_files(tmp_path: Path) -> None:
    rows = run_frontend_smoke_cases(fake_mode=True, output_dir=tmp_path)
    assert len(rows) == 5
    assert all("case" in item for item in rows)
    assert (tmp_path / "phase7r_step4a_agent_cockpit_smoke.json").exists()
    assert (tmp_path / "phase7r_step4a_agent_cockpit_smoke.md").exists()

    payload = json.loads((tmp_path / "phase7r_step4a_agent_cockpit_smoke.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 5


def test_smoke_single_case_error_does_not_break_others(tmp_path: Path) -> None:
    calls = {"n": 0}

    def _fake_run(_question: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first case failed")
        return {
            "result": {"run_id": "r", "session_id": "s", "metadata": {}, "workflow_decision": {}},
            "error": None,
            "logs": [],
            "elapsed_ms": 10,
        }

    cases = [
        {"name": "c1", "question": "q1"},
        {"name": "c2", "question": "q2"},
    ]
    with patch("self_ai.frontend_smoke.run_frontend_workflow_once", side_effect=_fake_run):
        rows = run_frontend_smoke_cases(cases=cases, fake_mode=False, output_dir=tmp_path)

    assert len(rows) == 2
    assert rows[0]["success"] is False
    assert rows[1]["success"] in {True, False}
