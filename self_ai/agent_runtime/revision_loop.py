# coding=utf-8
"""Storage-driven review + quality-gate + bounded revision loop."""

from __future__ import annotations

from typing import Any

from ..text_utils import shorten_text
from .agent_loop import AgentLoop
from .agent_policy import AgentPolicy
from .agent_session import AgentSession


class RevisionLoop:
    """Runs review/gate and optional bounded revisions."""

    def __init__(
        self,
        *,
        tool_runtime: Any,
        agent_loop: AgentLoop,
        policy: AgentPolicy,
        state_store: Any | None = None,
    ) -> None:
        self.tool_runtime = tool_runtime
        self.agent_loop = agent_loop
        self.policy = policy
        self.state_store = state_store

    async def _trace(
        self,
        session: AgentSession,
        event: str,
        *,
        run_context: Any | None = None,
        fields: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "event": event,
            "run_id": session.run_id,
            "session_id": session.session_id,
            "ts": session.updated_at,
            "fields": fields or {},
        }
        session.metadata.setdefault("trace_events", []).append(payload)
        if self.state_store is not None:
            await self.state_store.append_trace(session.run_id, payload, run_context=run_context)

    async def run_review(self, session: AgentSession, *, run_context: Any | None = None) -> dict[str, Any]:
        if not self.policy.should_review(session):
            return {}
        review_input = {
            "run_id": session.run_id,
            "session_id": session.session_id,
            "task": session.task,
            "task_profile": session.task_profile,
            "workflow_decision": session.workflow_decision,
            "plan": session.plan,
            "context_pack": session.context_pack,
            "agent_outputs": session.agent_outputs,
            "redis_context": {},
            "qdrant_review_memory": [],
            "neo4j_graph_context": [],
            "metadata": {
                "policy_suggested_agents": session.workflow_decision.get("selected_agents", []),
                "executed_role_agents": session.selected_agents,
                "revision_count": session.revision_count,
            },
        }
        result = await self.tool_runtime.execute_by_name(
            "review.run",
            {"session": session.model_dump(mode="json"), "review_input": review_input},
            run_id=session.run_id,
            session_id=session.session_id,
            run_context=run_context,
            metadata={"component": "agent_runtime", "stage": "review"},
        )
        if not result.ok:
            fallback = {
                "run_id": session.run_id,
                "session_id": session.session_id,
                "pass_review": False,
                "severity": "high",
                "major_issues": ["review_tool_failed"],
                "minor_issues": [],
                "missing_requirements": [],
                "requires_revision": True,
                "revision_target_agent": "synthesizer",
                "revision_instruction": "Review failed to execute, improve coherence and requirement coverage.",
                "evidence_refs": [],
                "storage_evidence": [],
                "rationale": "review_tool_failure_fallback",
                "metadata": {"error_type": (result.error or {}).get("type", "ReviewError")},
            }
            session.add_review_report(fallback)
            session.add_error(
                (result.error or {}).get("type", "ReviewError"),
                (result.error or {}).get("message", "review failed"),
                stage="revision_loop.review",
            )
            return fallback
        report = result.data.get("review_report", {})
        if not isinstance(report, dict):
            report = {
                "run_id": session.run_id,
                "session_id": session.session_id,
                "pass_review": False,
                "severity": "high",
                "major_issues": ["invalid_review_payload"],
                "minor_issues": [],
                "missing_requirements": [],
                "requires_revision": True,
                "revision_target_agent": "synthesizer",
                "revision_instruction": "Invalid review payload, rerun synthesis.",
                "evidence_refs": [],
                "storage_evidence": [],
                "rationale": "review_payload_fallback",
                "metadata": {"error_type": "InvalidReviewPayload"},
            }
        session.add_review_report(report)
        if self.state_store is not None:
            await self.state_store.save_node_output(
                session.run_id,
                "review",
                {
                    "pass_review": bool(report.get("pass_review", False)),
                    "severity": report.get("severity", ""),
                    "major_issue_count": len(report.get("major_issues", []))
                    if isinstance(report.get("major_issues"), list)
                    else 0,
                    "missing_requirement_count": len(report.get("missing_requirements", []))
                    if isinstance(report.get("missing_requirements"), list)
                    else 0,
                },
                run_context=run_context,
            )
        return report

    async def run_quality_gate(self, session: AgentSession, *, run_context: Any | None = None) -> dict[str, Any]:
        if not self.policy.should_run_quality_gate(session):
            return {}
        latest = session.review_reports[-1] if session.review_reports else {}
        if not isinstance(latest, dict):
            latest = {}
        result = await self.tool_runtime.execute_by_name(
            "review.quality_gate.decide",
            {
                "review_report": latest,
                "task_profile": session.task_profile,
                "run_id": session.run_id,
                "session_id": session.session_id,
                "max_revision_iterations": session.max_revision_iterations,
                "current_iteration": session.revision_count,
            },
            run_id=session.run_id,
            session_id=session.session_id,
            run_context=run_context,
            metadata={"component": "agent_runtime", "stage": "quality_gate"},
        )
        if not result.ok:
            fallback_gate = {
                "run_id": session.run_id,
                "session_id": session.session_id,
                "passed": False,
                "decision": "warn",
                "severity": "medium",
                "requires_revision": False,
                "revision_target_agent": None,
                "revision_instruction": "",
                "max_revision_iterations": session.max_revision_iterations,
                "current_iteration": session.revision_count,
                "blocking_issues": [],
                "warnings": ["quality_gate_tool_failure"],
                "rationale": "quality_gate_tool_failure_fallback",
                "metadata": {"error_type": (result.error or {}).get("type", "QualityGateError")},
            }
            session.set_quality_gate(fallback_gate)
            session.add_error(
                (result.error or {}).get("type", "QualityGateError"),
                (result.error or {}).get("message", "quality gate failed"),
                stage="revision_loop.quality_gate",
            )
            return fallback_gate
        gate = result.data.get("quality_gate", {})
        if not isinstance(gate, dict):
            gate = {"decision": "warn", "passed": True, "requires_revision": False}
        session.set_quality_gate(gate)
        if self.state_store is not None:
            await self.state_store.save_node_output(
                session.run_id,
                "quality_gate",
                {
                    "decision": gate.get("decision"),
                    "passed": bool(gate.get("passed", False)),
                    "requires_revision": bool(gate.get("requires_revision", False)),
                    "severity": gate.get("severity", ""),
                },
                run_context=run_context,
            )
        return gate

    def should_continue_revision(self, session: AgentSession) -> bool:
        if not self.policy.should_revise(session):
            return False
        return session.revision_count < session.max_revision_iterations

    async def run_revision_iteration(
        self,
        session: AgentSession,
        *,
        run_context: Any | None = None,
        task_board: Any | None = None,
        message_bus: Any | None = None,
    ) -> None:
        target = self.policy.select_revision_target(session)
        if not target:
            session.add_error("RevisionTargetMissing", "no revision target agent", stage="revision_loop")
            return
        if target not in set(session.selected_agents + ["synthesizer"]):
            session.add_error("RevisionTargetInvalid", f"target {target} not in selected agents", stage="revision_loop")
            return
        session.current_target_agent = target
        gate = session.quality_gate if isinstance(session.quality_gate, dict) else {}
        instruction = str(gate.get("revision_instruction", "") or "").strip()
        if not instruction and session.review_reports and isinstance(session.review_reports[-1], dict):
            instruction = str(session.review_reports[-1].get("revision_instruction", "") or "").strip()
        if not instruction:
            instruction = "Address major issues and missing requirements from review."

        if task_board is not None:
            task_board.create_task(
                goal=f"Revision iteration #{session.revision_count} for {target}",
                agent_name=target,
                metadata={"revision_instruction": instruction[:300]},
            )
        if message_bus is not None:
            message_bus.send(
                sender="revision_loop",
                recipient=target,
                content=instruction,
                message_type="revision_instruction",
            )
        session.append_message(
            sender="revision_loop",
            recipient=target,
            content=instruction,
            message_type="revision_instruction",
        )

        await self.agent_loop.run_agent(
            session,
            agent_name=target,
            run_context=run_context,
            instruction=instruction,
        )
        if target != "synthesizer" and "synthesizer" in session.selected_agents:
            await self.agent_loop.run_synthesizer(
                session,
                run_context=run_context,
                instruction=f"Integrate revision from {target}: {shorten_text(instruction, max_chars=300)}",
            )
        await self.run_review(session, run_context=run_context)
        await self.run_quality_gate(session, run_context=run_context)

    async def run_until_complete(
        self,
        session: AgentSession,
        *,
        run_context: Any | None = None,
        task_board: Any | None = None,
        message_bus: Any | None = None,
    ) -> AgentSession:
        await self._trace(session, "revision_loop.start", run_context=run_context)
        await self.run_review(session, run_context=run_context)
        await self.run_quality_gate(session, run_context=run_context)

        while self.should_continue_revision(session):
            iteration = session.revision_count + 1
            session.revision_count = iteration
            session.touch()
            await self._trace(
                session,
                "revision_loop.iteration.start",
                run_context=run_context,
                fields={"iteration": iteration},
            )
            await self.run_revision_iteration(
                session,
                run_context=run_context,
                task_board=task_board,
                message_bus=message_bus,
            )
            await self._trace(
                session,
                "revision_loop.iteration.end",
                run_context=run_context,
                fields={
                    "iteration": iteration,
                    "decision": (session.quality_gate or {}).get("decision", ""),
                },
            )

        session.metadata["revision_count"] = session.revision_count
        session.metadata["max_revision_iterations"] = session.max_revision_iterations
        session.metadata["revision_performed"] = session.revision_count > 0
        session.metadata["revision_target_agent"] = session.current_target_agent
        await self._trace(
            session,
            "revision_loop.end",
            run_context=run_context,
            fields={
                "revision_count": session.revision_count,
                "max_revision_iterations": session.max_revision_iterations,
                "decision": (session.quality_gate or {}).get("decision", ""),
            },
        )
        return session

