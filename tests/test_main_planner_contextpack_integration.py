# coding=utf-8
"""MainLoop contract checks for plan/context fields."""

from pathlib import Path
from unittest.mock import patch

import pytest

from self_ai import main
from tests._engine_helpers import build_test_kernel


@pytest.mark.asyncio
async def test_architecture_design_returns_plan_and_context_pack_dicts(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Architecture design for a multi-agent runtime")

    assert isinstance(result["plan"], dict)
    assert isinstance(result["context_pack"], dict)
    assert result["response"]


@pytest.mark.asyncio
async def test_document_writing_keeps_contract_for_plan_and_context_pack(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Write a short project document introduction")

    assert isinstance(result["plan"], dict)
    assert isinstance(result["context_pack"], dict)


@pytest.mark.asyncio
async def test_research_summary_returns_context_pack_safely(tmp_path: Path) -> None:
    kernel, _ = build_test_kernel(tmp_path)
    with patch("self_ai.main.get_kernel", return_value=kernel):
        result = await main.run_autonomy_workflow("Create a research summary for memory components")

    assert isinstance(result["context_pack"], dict)
    assert isinstance(result["errors"], list)
