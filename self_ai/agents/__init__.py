# coding=utf-8
"""Role agents for Phase 6A."""

from .architect import ArchitectAgent
from .base import AgentInput, AgentResult, BaseRoleAgent
from .coder import CoderAgent
from .researcher import ResearcherAgent
from .reviewer import ReviewerAgent, ReviewerInput
from .selector import select_agents_for_state
from .synthesizer import SynthesizerAgent
from .writer import WriterAgent

__all__ = [
    "AgentInput",
    "AgentResult",
    "BaseRoleAgent",
    "CoderAgent",
    "ArchitectAgent",
    "ResearcherAgent",
    "WriterAgent",
    "SynthesizerAgent",
    "ReviewerInput",
    "ReviewerAgent",
    "select_agents_for_state",
]
