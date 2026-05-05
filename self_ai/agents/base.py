# coding=utf-8
"""Base contracts for Phase 6A role agents."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from ..text_utils import shorten_text


class AgentInput(BaseModel):
    run_id: str = ""
    session_id: str = "default"
    task: str = ""
    task_profile: dict[str, Any] = Field(default_factory=dict)
    workflow_decision: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    context_pack: dict[str, Any] = Field(default_factory=dict)
    research_results: list[dict[str, Any]] = Field(default_factory=list)
    debate_messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    run_id: str = ""
    session_id: str = "default"
    agent_name: str
    role: str
    output: str = ""
    output_type: str = "text"
    summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    used_context: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseRoleAgent:
    """Base role agent abstraction with safe execution fallback."""

    name: str = "base"
    role: str = "base"
    model_category: str = "fast"
    output_type: str = "text"

    def build_prompt(self, agent_input: AgentInput) -> str:
        task = shorten_text(agent_input.task, max_chars=500)
        return (
            f"You are {self.__class__.__name__}.\n"
            f"Role: {self.role}\n"
            "Return concise output aligned with your role.\n\n"
            f"Task:\n{task}\n"
        )

    def _build_used_context(self, agent_input: AgentInput) -> list[str]:
        used: list[str] = []
        if agent_input.context_pack:
            used.append("context_pack")
        if agent_input.plan:
            used.append("plan")
        if agent_input.research_results:
            used.append("research_results")
        if agent_input.debate_messages:
            used.append("debate_messages")
        return used

    async def run(
        self,
        agent_input: AgentInput,
        *,
        route_model_func: Callable[[str, str], Awaitable[dict[str, Any]]] | None = None,
    ) -> AgentResult:
        """Run the role agent with safe fallback behavior."""
        if route_model_func is None:
            from ..router import route_model as route_model_func  # lazy import

        prompt = self.build_prompt(agent_input)
        used_context = self._build_used_context(agent_input)
        try:
            raw = await route_model_func(prompt, self.model_category)
            output = str(raw.get("response", "") or "").strip()
            return AgentResult(
                run_id=agent_input.run_id,
                session_id=agent_input.session_id,
                agent_name=self.name,
                role=self.role,
                output=output,
                output_type=self.output_type,
                summary=shorten_text(output, max_chars=180),
                confidence=0.7 if output else 0.35,
                used_context=used_context,
                metadata={
                    "model": raw.get("model", ""),
                    "model_category": self.model_category,
                },
            )
        except Exception as exc:  # pragma: no cover - exercised via unit tests
            return AgentResult(
                run_id=agent_input.run_id,
                session_id=agent_input.session_id,
                agent_name=self.name,
                role=self.role,
                output="",
                output_type=self.output_type,
                summary=f"{self.name} unavailable",
                confidence=0.2,
                used_context=used_context,
                warnings=[f"model_error:{type(exc).__name__}"],
                metadata={
                    "error": type(exc).__name__,
                    "model_category": self.model_category,
                },
            )

