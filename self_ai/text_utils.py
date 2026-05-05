# coding=utf-8
"""Text normalization and surface signal extraction for Self AI."""

import re
from typing import Final
import warnings

from .schemas import SurfaceSignals

_CONTROL_CHARS_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_CJK_RE: Final[re.Pattern[str]] = re.compile(r"[\u4e00-\u9fff]")
_CODE_BLOCK_RE: Final[re.Pattern[str]] = re.compile(r"```[\s\S]*?```")
_FILE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"([A-Za-z]:\\|/[\w\-.]+/|[\w\-.]+\.(py|md|json|yaml|yml|toml|js|ts)\b)"
)
_ERROR_TRACE_RE: Final[re.Pattern[str]] = re.compile(
    r"(Traceback \(most recent call last\)|Exception|Error:|TypeError|ValueError|RuntimeError)"
)
_URL_RE: Final[re.Pattern[str]] = re.compile(r"https?://\S+")
_MARKDOWN_TABLE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_PATH_CANDIDATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'`]+|(?:\./|\.\./|/)?[^\s\"'`]+(?:/[^\s\"'`]+)*\.[A-Za-z0-9]{1,12})"
)
_SUSPICIOUS_MOJIBAKE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:Ã.|â.|å.|æ.|ç.|ð.|ä.|ï.)"
)


def _count_cjk(value: str) -> int:
    return len(_CJK_RE.findall(value))


def _maybe_repair_mojibake(value: str) -> str:
    """Best-effort mojibake repair for common UTF-8 mis-decoding paths."""
    if not value:
        return value
    if _count_cjk(value) > 0 and "�" not in value:
        return value
    if not _SUSPICIOUS_MOJIBAKE_RE.search(value):
        return value

    candidates: list[str] = [value]
    for src_encoding in ("latin-1", "cp1252"):
        try:
            repaired = value.encode(src_encoding, errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                candidates.append(repaired)
        except Exception:
            continue

    def _score(text: str) -> tuple[int, int, int]:
        cjk = _count_cjk(text)
        replacement = text.count("�")
        qmarks = text.count("?")
        # More CJK is better; fewer replacement and '?' are better.
        return (cjk, -replacement, -qmarks)

    return max(candidates, key=_score)


def normalize_text(value: str) -> str:
    """Normalize input text while preserving language semantics."""
    repaired = _maybe_repair_mojibake(value)
    cleaned = _CONTROL_CHARS_RE.sub("", repaired)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def is_probably_garbled(value: str) -> bool:
    """Detect suspicious mojibake-like input dominated by question marks."""
    if not value:
        return False
    qmarks = value.count("?")
    replacement = value.count("�")
    ratio = qmarks / max(len(value), 1)
    cjk_count = len(_CJK_RE.findall(value))
    return (ratio >= 0.25 and cjk_count == 0 and len(value) >= 8) or (replacement >= 3 and cjk_count == 0)


def extract_path_candidates(value: str, *, max_items: int = 8) -> list[str]:
    """Extract likely file path candidates from free-form text."""
    items: list[str] = []
    for match in _PATH_CANDIDATE_RE.findall(value or ""):
        candidate = str(match).strip().strip("\"'`")
        if not candidate:
            continue
        normalized = candidate.replace("\\", "/")
        if normalized not in items:
            items.append(normalized)
        if len(items) >= max_items:
            break
    return items


def extract_surface_signals(value: str) -> SurfaceSignals:
    """Extract lightweight observable signals from input text."""
    return SurfaceSignals(
        has_code_block=bool(_CODE_BLOCK_RE.search(value)),
        has_file_path=bool(_FILE_PATH_RE.search(value)),
        has_error_trace=bool(_ERROR_TRACE_RE.search(value)),
        has_url=bool(_URL_RE.search(value)),
        has_chinese=bool(_CJK_RE.search(value)),
        has_markdown_table=bool(_MARKDOWN_TABLE_RE.search(value)),
        length=len(value),
    )


def is_coding_task(value: str) -> bool:
    """Heuristic for coding-oriented tasks.

    Deprecated: use task_analyzer.analyze_task instead.
    """
    warnings.warn(
        "is_coding_task is deprecated. Use task_analyzer.analyze_task instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    lower = value.lower()
    keywords = [
        "python",
        "code",
        "function",
        "api",
        "bug",
        "fix",
        "refactor",
        "算法",
        "代码",
        "函数",
        "接口",
        "调试",
        "修复",
        "重构",
    ]
    return any(k in lower for k in keywords)


def is_complex_task(value: str) -> bool:
    """Heuristic for tasks that need extended reasoning/debate.

    Deprecated: use task_analyzer.analyze_task instead.
    """
    warnings.warn(
        "is_complex_task is deprecated. Use task_analyzer.analyze_task instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    lower = value.lower()
    keywords = [
        "complex",
        "architecture",
        "tradeoff",
        "distributed",
        "microservice",
        "多方案",
        "架构",
        "权衡",
        "复杂",
        "系统设计",
        "性能优化",
    ]
    return len(value) > 120 or any(k in lower for k in keywords)


def shorten_text(value: str, max_chars: int = 600) -> str:
    """Truncate text for prompt moderation context."""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."
