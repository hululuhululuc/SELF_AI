# coding=utf-8
"""Unit tests for text utilities."""

import warnings

from self_ai.text_utils import (
    extract_path_candidates,
    extract_surface_signals,
    is_complex_task,
    is_coding_task,
    is_probably_garbled,
    normalize_text,
)


def test_normalize_text_removes_control_and_normalizes_whitespace() -> None:
    raw = "hello\x00  world\t\n  test"
    assert normalize_text(raw) == "hello world test"


def test_is_probably_garbled_with_many_question_marks() -> None:
    assert is_probably_garbled("?????????????abcde")
    assert not is_probably_garbled("这是正常中文输入")


def test_extract_surface_signals_detects_expected_patterns() -> None:
    text = (
        "Traceback (most recent call last)\n"
        "See file main.py and docs/readme.md\n"
        "https://example.com\n"
        "| a | b |\n"
        "```python\nprint('x')\n```\n"
        "中文内容"
    )
    signals = extract_surface_signals(text)
    assert signals.has_code_block
    assert signals.has_file_path
    assert signals.has_error_trace
    assert signals.has_url
    assert signals.has_markdown_table
    assert signals.has_chinese
    assert signals.length == len(text)


def test_legacy_helpers_emit_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        _ = is_coding_task("write python code")
        _ = is_complex_task("complex architecture tradeoff")
    categories = [r.category for r in records]
    assert DeprecationWarning in categories


def test_normalize_text_repairs_common_mojibake() -> None:
    mojibake = "ä½ å¥½, this is broken utf8 decode"
    repaired = normalize_text(mojibake)
    assert "你好" in repaired


def test_extract_path_candidates_finds_multiple_paths() -> None:
    text = "Please create artifacts/demo.txt and then edit src/app.py"
    paths = extract_path_candidates(text)
    assert "artifacts/demo.txt" in paths
    assert "src/app.py" in paths
