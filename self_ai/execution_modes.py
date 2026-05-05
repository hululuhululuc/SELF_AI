# coding=utf-8
"""Execution mode definitions for workflow policy routing."""

from enum import StrEnum


class ExecutionMode(StrEnum):
    QUICK_ANSWER = "quick_answer"
    RESEARCH_THEN_IMPLEMENT = "research_then_implement"
    CODE_FOCUSED = "code_focused"
    ARCHITECTURE_DESIGN = "architecture_design"
    DOCUMENT_WRITING = "document_writing"
    GENERAL = "general"
