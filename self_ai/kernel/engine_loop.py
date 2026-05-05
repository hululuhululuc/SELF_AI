# coding=utf-8
"""MainLoop executor: model-centric loop + tool dispatch."""

from __future__ import annotations

import json
import re
from typing import Any

from .engine_state import EngineState
from .result_aggregator import ResultAggregator
from .state_store import StateStore
from ..text_utils import extract_path_candidates

_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)```|```\s*([\s\S]*?)```")
_MUTATION_TOOLS = {
    "workspace.file.create",
    "workspace.file.write",
    "workspace.file.edit",
    "workspace.file.rename",
    "workspace.file.delete",
    "storage.semantic.upsert_evidence",
    "storage.semantic.upsert_memory",
    "storage.graph.record_issue_fix",
}
_VERIFICATION_TOOLS = {
    "workspace.file.read",
    "workspace.file.list",
}
_READ_EVIDENCE_PERMISSIONS = {
    "read_only",
    "workspace_read",
    "memory_read",
}
_VERIFICATION_DUP_WINDOW = 12
_MAINLOOP_HIDDEN_TOOLS = {
    "storage.runtime.append_trace",
    "storage.runtime.save_node_output",
    "storage.runtime.save_agent_output",
    "storage.runtime.save_run_state",
    "storage.runtime.load_run_state",
    "storage.semantic.upsert_evidence",
    "storage.semantic.upsert_memory",
}
_PATH_CONTINUATION_CUES_RE = re.compile(
    r"\b(still|same|that|those|again|final)\b|最终|仍然|同一个|该文件|这个文件|上一|上述",
    re.IGNORECASE,
)


class EngineLoop:
    """Single-loop runtime.

    The model decides each turn:
    - final_answer
    - tool_calls
    System only enforces safety, executes tools, and aggregates state.
    """

    def __init__(
        self,
        *,
        tool_runtime: Any,
        state_store: StateStore,
        prompt_runtime: Any | None = None,
        result_aggregator: ResultAggregator | None = None,
        agent_runtime: Any | None = None,
    ) -> None:
        self.tool_runtime = tool_runtime
        self.state_store = state_store
        self.prompt_runtime = prompt_runtime
        self.result_aggregator = result_aggregator or ResultAggregator()
        self.agent_runtime = agent_runtime

    async def _trace(
        self,
        state: EngineState,
        event: str,
        *,
        stage: str | None = None,
        status: str | None = None,
        fields: dict[str, Any] | None = None,
        run_context: Any | None = None,
    ) -> None:
        state.append_trace(event, stage=stage, status=status, fields=fields or {})
        if state.trace:
            await self.state_store.append_trace(
                state.run_id,
                state.trace[-1],
                run_context=run_context,
            )

    async def _call_tool(
        self,
        state: EngineState,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        stage: str,
        run_context: Any | None = None,
    ) -> Any:
        await self._trace(
            state,
            "engine.stage.tool.start",
            stage=stage,
            status="start",
            fields={"tool": tool_name},
            run_context=run_context,
        )
        result = await self.tool_runtime.execute_by_name(
            tool_name,
            arguments,
            run_id=state.run_id,
            session_id=state.session_id,
            run_context=run_context,
            metadata={"stage": stage},
        )
        self.result_aggregator.merge_tool_result(state, result, stage)
        await self._trace(
            state,
            "engine.stage.tool.end",
            stage=stage,
            status="end" if result.ok else "error",
            fields={
                "tool": tool_name,
                "ok": result.ok,
                "latency_ms": result.latency_ms,
            },
            run_context=run_context,
        )
        return result

    async def _save_stage_output(
        self,
        state: EngineState,
        stage: str,
        output: dict[str, Any],
        *,
        run_context: Any | None = None,
    ) -> None:
        await self.state_store.save_node_output(
            state.run_id,
            stage,
            output,
            run_context=run_context,
        )
        await self.state_store.save_state(state, run_context=run_context)

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        stripped = (text or "").strip()
        if not stripped:
            raise ValueError("empty_json")
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = _JSON_BLOCK_RE.search(stripped)
        if fenced:
            candidate = fenced.group(1) or fenced.group(2) or ""
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("invalid_json")

    def _tool_catalog(self, run_context: Any | None = None) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for spec in self.tool_runtime.registry.list():
            if spec.name == "model.generate":
                continue
            if spec.name in _MAINLOOP_HIDDEN_TOOLS:
                continue
            allowed = self.tool_runtime.permission_guard.can(
                spec.permission,
                run_context=run_context,
            )
            if not allowed:
                continue
            tools.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "permission": spec.permission,
                    "timeout_s": spec.timeout_s,
                }
            )
        return tools

    @staticmethod
    def _safe_json(value: Any, *, max_len: int = 3000) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= max_len else text[:max_len] + "...(truncated)"

    def _estimate_context_tokens(self, state: EngineState) -> int:
        snapshot = {
            "task": state.normalized_task or state.task,
            "errors": state.errors[-12:],
            "recent_tool_results": state.metadata.get("recent_tool_results", [])[-20:],
            "execution_state": state.metadata.get("execution_state", {}),
            "goal_contract": state.metadata.get("goal_contract", {}),
        }
        return max(1, len(json.dumps(snapshot, ensure_ascii=False, default=str)) // 4)

    def _microcompact_recent_results(self, state: EngineState) -> None:
        recent = state.metadata.get("recent_tool_results", [])
        if not isinstance(recent, list):
            return
        changed = False
        for item in recent:
            if not isinstance(item, dict):
                continue
            preview = str(item.get("data_preview", ""))
            if len(preview) > 360:
                item["data_preview"] = preview[:360] + "...(microcompact)"
                changed = True
            arg_preview = str(item.get("arguments_preview", ""))
            if len(arg_preview) > 240:
                item["arguments_preview"] = arg_preview[:240] + "...(microcompact)"
                changed = True
        if changed:
            state.metadata["recent_tool_results"] = recent
            state.metadata["context_compaction"] = {
                "mode": "micro",
                "recent_tool_results_count": len(recent),
            }

    def _auto_compact_recent_results(self, state: EngineState) -> None:
        recent = state.metadata.get("recent_tool_results", [])
        if not isinstance(recent, list) or len(recent) <= 8:
            return
        keep = recent[-8:]
        summary = {
            "dropped_count": len(recent) - len(keep),
            "ok_count": sum(1 for x in recent if isinstance(x, dict) and bool(x.get("ok", False))),
            "fail_count": sum(1 for x in recent if isinstance(x, dict) and not bool(x.get("ok", False))),
            "tool_histogram": {},
        }
        hist: dict[str, int] = {}
        for item in recent:
            if not isinstance(item, dict):
                continue
            name = str(item.get("tool_name", "")).strip()
            if not name:
                continue
            hist[name] = hist.get(name, 0) + 1
        summary["tool_histogram"] = hist
        state.metadata["recent_tool_results"] = keep
        state.metadata["context_compaction"] = {
            "mode": "auto",
            "recent_tool_results_count": len(keep),
            "summary": summary,
        }

    def _derive_goal_contract(self, state: EngineState) -> dict[str, Any]:
        task = str(state.normalized_task or state.task or "")
        target_paths = self._resolve_target_paths_with_history(state, extract_path_candidates(task))
        source = "derived_from_task_shape"
        if target_paths:
            source = "derived_from_task_shape+history_anchor"
        contract = {
            "requires_side_effect": False,
            "requires_read_proof": False,
            "target_paths": target_paths[:6],
            "has_explicit_target_path": bool(target_paths),
            "intent_kind": "unknown",
            "source": source,
        }
        return contract

    def _resolve_target_paths_with_history(self, state: EngineState, extracted_paths: list[str]) -> list[str]:
        normalized = [self._normalize_path(p) for p in extracted_paths if str(p).strip()]
        hints = self._collect_history_path_hints(state)
        if not hints:
            return normalized[:6]

        resolved: list[str] = []
        for raw in normalized:
            candidate = self._resolve_relative_target_with_history(raw, hints)
            if candidate and candidate not in resolved:
                resolved.append(candidate)
            # If a basename path was uniquely resolved to a concrete anchored
            # path from session history, keep the resolved path only.
            keep_raw = True
            if candidate and candidate != raw and "/" not in raw and ":" not in raw:
                keep_raw = False
            if keep_raw and raw and raw not in resolved:
                resolved.append(raw)
        if resolved:
            return resolved[:6]

        task = str(state.normalized_task or state.task or "")
        if not _PATH_CONTINUATION_CUES_RE.search(task):
            return normalized[:6]

        preferred = self._pick_history_anchor_for_task(task=task, hints=hints)
        if preferred:
            return [preferred]
        return normalized[:6]

    def _collect_history_path_hints(self, state: EngineState) -> list[str]:
        hints: list[str] = []
        metadata = state.metadata if isinstance(state.metadata, dict) else {}

        def _add_path(path_text: Any) -> None:
            normalized = self._normalize_path(path_text)
            if normalized and normalized not in hints:
                hints.append(normalized)

        recent_tool_results = metadata.get("recent_tool_results", [])
        if isinstance(recent_tool_results, list):
            for row in reversed(recent_tool_results[-12:]):
                if not isinstance(row, dict):
                    continue
                args = row.get("arguments_preview", "")
                data_preview = row.get("data_preview", "")
                for source_text in (args, data_preview):
                    for extracted in extract_path_candidates(str(source_text or ""), max_items=8):
                        _add_path(extracted)

        recent_turns = metadata.get("chat_recent_turns", [])
        if isinstance(recent_turns, list):
            for row in reversed(recent_turns[-12:]):
                if not isinstance(row, dict):
                    continue
                for key in ("preview", "task_preview", "response_preview", "retrieval_summary", "content_preview"):
                    value = row.get(key, "")
                    if not value:
                        continue
                    for extracted in extract_path_candidates(str(value), max_items=8):
                        _add_path(extracted)

        return hints[:24]

    @staticmethod
    def _path_basename(path_text: str) -> str:
        text = str(path_text or "").strip().replace("\\", "/")
        if not text:
            return ""
        return text.rsplit("/", 1)[-1]

    def _resolve_relative_target_with_history(self, target: str, hints: list[str]) -> str:
        normalized = self._normalize_path(target)
        if not normalized:
            return ""
        if "/" in normalized or normalized.startswith(".") or ":" in normalized:
            return normalized
        basename = self._path_basename(normalized)
        matched = [hint for hint in hints if self._path_basename(hint) == basename and "/" in hint]
        if len(matched) == 1:
            return matched[0]
        return normalized

    def _pick_history_anchor_for_task(self, *, task: str, hints: list[str]) -> str:
        lowered = task.lower()
        if "final" in lowered or "最终" in task:
            for hint in hints:
                base = self._path_basename(hint).lower()
                if "final" in base:
                    return hint
        for hint in hints:
            if "." in self._path_basename(hint):
                return hint
        return hints[0] if hints else ""

    def _build_goal_contract_prompt(self, state: EngineState) -> str:
        task = str(state.normalized_task or state.task or "")
        derived = state.metadata.get("goal_contract", {})
        return (
            "You are a goal-contract classifier for an agentic execution loop.\n"
            "Decide task intent and proof requirements. Return JSON only.\n"
            "Schema:\n"
            "{\n"
            '  "intent_kind": "state_change|read_only|analysis",\n'
            '  "requires_side_effect": true|false,\n'
            '  "requires_read_proof": true|false,\n'
            '  "target_paths": ["optional/path"],\n'
            '  "rationale": "short reason"\n'
            "}\n"
            "Rules:\n"
            "- state_change means file/system state must be changed.\n"
            "- read_only means no mutation, but response must be grounded in read evidence.\n"
            "- analysis means conceptual answer; read proof optional unless task asks to inspect project files.\n"
            "- Conversational memory requests (remember/apply a user preference, style, fact, or future instruction) "
            "are NOT workspace state changes; runtime/sidecar handles chat memory persistence.\n"
            "- If task explicitly requests create/edit/rename/delete/write/rewrite/append/modify, requires_side_effect MUST be true.\n"
            "- Set requires_side_effect true only for explicit workspace/file/system mutations requested by the user; "
            "do not invent files or paths to persist conversation memory.\n"
            "- If unsure, be conservative and set requires_side_effect=true only when mutation is explicitly requested.\n"
            "Examples:\n"
            "Q: Create a.py and write content into it.\n"
            'A: {"intent_kind":"state_change","requires_side_effect":true,"requires_read_proof":false,"target_paths":["a.py"]}\n'
            "Q: Delete logs/app.log.\n"
            'A: {"intent_kind":"state_change","requires_side_effect":true,"requires_read_proof":false,"target_paths":["logs/app.log"]}\n'
            "Q: Read README.md and summarize it.\n"
            'A: {"intent_kind":"read_only","requires_side_effect":false,"requires_read_proof":true,"target_paths":["README.md"]}\n'
            "Q: Explain the role boundary of Redis and Qdrant.\n"
            'A: {"intent_kind":"analysis","requires_side_effect":false,"requires_read_proof":false,"target_paths":[]}\n'
            "Q: In future replies, use concise bullet points and remember that preference.\n"
            'A: {"intent_kind":"analysis","requires_side_effect":false,"requires_read_proof":false,"target_paths":[]}\n'
            f"Task:\n{task}\n"
            f"Derived hints:\n{self._safe_json(derived, max_len=500)}\n"
        )

    def _build_goal_side_effect_verify_prompt(self, state: EngineState) -> str:
        task = str(state.normalized_task or state.task or "")
        return (
            "You are a verifier.\n"
            "Question: Does completing the user task require mutating workspace state (create/write/edit/rename/delete)?\n"
            "Return JSON only:\n"
            '{ "requires_side_effect": true|false, "confidence": 0.0-1.0, "reason": "short" }\n'
            "Examples:\n"
            "Q: Delete a.txt -> true\n"
            "Q: Create src/a.py and write code -> true\n"
            "Q: Read README and summarize -> false\n"
            f"Task:\n{task}\n"
        )

    async def _infer_side_effect_vote_by_model(self, state: EngineState, *, run_context: Any | None = None) -> bool | None:
        result = await self._call_tool(
            state,
            tool_name="model.generate",
            arguments={
                "prompt": self._build_goal_side_effect_verify_prompt(state),
                "stage": "mainloop_goal_contract_verify",
                "hints": {
                    "goal": "side_effect_binary_json",
                    "prefer_structured_json": True,
                    "force_tier": "fast",
                    "enable_thinking": False,
                    "max_tokens": 512,
                },
            },
            stage="goal_contract_verify",
            run_context=run_context,
        )
        if not result.ok:
            return None
        data = result.data if isinstance(result.data, dict) else {}
        raw = str(data.get("response", "") or "")
        try:
            parsed = self._extract_json_object(raw)
        except Exception:
            return None
        if "requires_side_effect" not in parsed:
            return None
        return bool(parsed.get("requires_side_effect", False))

    async def _infer_goal_contract_by_model(
        self,
        state: EngineState,
        *,
        run_context: Any | None = None,
        enable_verify_vote: bool = False,
    ) -> None:
        prompt = self._build_goal_contract_prompt(state)
        result = await self._call_tool(
            state,
            tool_name="model.generate",
            arguments={
                "prompt": prompt,
                "stage": "mainloop_goal_contract",
                "hints": {
                    "goal": "goal_contract_json",
                    "prefer_structured_json": True,
                    "force_tier": "fast",
                    "enable_thinking": False,
                    "max_tokens": 2048,
                },
            },
            stage="goal_contract_infer",
            run_context=run_context,
        )
        if not result.ok:
            state.add_error(
                "GoalContractInferFailed",
                str((result.error or {}).get("message", "model call failed")),
                stage="goal_contract_infer",
            )
            await self._trace(
                state,
                "goal_contract.infer.error",
                stage="goal_contract_infer",
                status="error",
                fields={"error": (result.error or {}).get("type", "ToolError")},
                run_context=run_context,
            )
            return
        data = result.data if isinstance(result.data, dict) else {}
        raw = str(data.get("response", "") or "")
        parsed = self._extract_json_object(raw)
        intent_kind = str(parsed.get("intent_kind", "")).strip().lower()
        if intent_kind not in {"state_change", "read_only", "analysis"}:
            intent_kind = "unknown"
        current = state.metadata.get("goal_contract", {})
        if not isinstance(current, dict):
            current = self._derive_goal_contract(state)
        merged = dict(current)
        merged["intent_kind"] = intent_kind
        # Proof requirements are monotonic. Model can raise strictness but must
        # not silently lower previously-required proof constraints.
        merged["requires_side_effect"] = bool(current.get("requires_side_effect", False)) or bool(
            parsed.get("requires_side_effect", False)
        )
        merged["requires_read_proof"] = bool(current.get("requires_read_proof", False)) or bool(
            parsed.get("requires_read_proof", False)
        )
        parsed_paths = parsed.get("target_paths", [])
        if isinstance(parsed_paths, list):
            merged_paths = [self._normalize_path(p) for p in merged.get("target_paths", []) if str(p).strip()]
            for raw_path in parsed_paths:
                norm = self._normalize_path(raw_path)
                if norm and norm not in merged_paths:
                    merged_paths.append(norm)
            merged["target_paths"] = merged_paths[:6]
            merged["has_explicit_target_path"] = bool(merged["target_paths"])
        merged["source"] = "model_goal_contract"
        merged["rationale"] = str(parsed.get("rationale", "") or "")[:220]
        if enable_verify_vote and bool(merged.get("has_explicit_target_path", False)):
            vote = await self._infer_side_effect_vote_by_model(state, run_context=run_context)
            if vote is True:
                merged["requires_side_effect"] = True
                if str(merged.get("intent_kind", "unknown")) in {"unknown", "read_only", "analysis"}:
                    merged["intent_kind"] = "state_change"
                merged["source"] = "model_goal_contract+verify_vote"
        state.metadata["goal_contract"] = merged
        await self._trace(
            state,
            "goal_contract.infer.end",
            stage="goal_contract_infer",
            status="ok",
            fields={
                "intent_kind": merged.get("intent_kind", "unknown"),
                "requires_side_effect": bool(merged.get("requires_side_effect", False)),
                "requires_read_proof": bool(merged.get("requires_read_proof", False)),
                "target_path_count": len(merged.get("target_paths", [])),
            },
            run_context=run_context,
        )

    @staticmethod
    def _normalize_path(value: Any) -> str:
        text = str(value or "").strip().strip("\"'`")
        return text.replace("\\", "/")

    def _extract_call_paths(self, arguments: dict[str, Any], data: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for key in ("path", "from_path", "to_path", "cwd"):
            if key in arguments:
                normalized = self._normalize_path(arguments.get(key))
                if normalized and normalized not in paths:
                    paths.append(normalized)
        data_path = self._normalize_path(data.get("path"))
        if data_path and data_path not in paths:
            paths.append(data_path)
        return paths

    @staticmethod
    def _workspace_version(state: EngineState) -> int:
        return int(state.metadata.get("workspace_snapshot_version", 0) or 0)

    def _bump_workspace_version(self, state: EngineState) -> int:
        new_version = self._workspace_version(state) + 1
        state.metadata["workspace_snapshot_version"] = new_version
        return new_version

    def _semantic_signature(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        data: dict[str, Any],
    ) -> str:
        normalized_paths = sorted(self._extract_call_paths(arguments, data))
        if tool_name == "workspace.file.list":
            recursive = bool(arguments.get("recursive", False))
            base_path = self._normalize_path(data.get("base_path") or arguments.get("path") or ".")
            return f"{tool_name}|path={base_path}|recursive={int(recursive)}"
        if tool_name == "workspace.file.read":
            path = self._normalize_path(data.get("path") or arguments.get("path") or "")
            return f"{tool_name}|path={path}"
        if tool_name in _MUTATION_TOOLS:
            return f"{tool_name}|paths={','.join(normalized_paths)}"
        payload = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
        return f"{tool_name}|args={payload}"

    def _update_workspace_snapshot(
        self,
        state: EngineState,
        *,
        tool_name: str,
        ok: bool,
        data: dict[str, Any],
        arguments: dict[str, Any],
        execution_flag: dict[str, Any],
        semantic_signature: str,
    ) -> None:
        snapshot = state.metadata.get("workspace_snapshot", {})
        if not isinstance(snapshot, dict):
            snapshot = {}
        reads = snapshot.get("reads", {})
        if not isinstance(reads, dict):
            reads = {}
        lists = snapshot.get("lists", {})
        if not isinstance(lists, dict):
            lists = {}

        current_version = self._workspace_version(state)
        if tool_name == "workspace.file.read" and ok:
            path = self._normalize_path(data.get("path") or arguments.get("path") or "")
            if path:
                reads[path] = {
                    "semantic_signature": semantic_signature,
                    "snapshot_id": str(data.get("snapshot_id", "") or ""),
                    "content_hash": str(data.get("content_hash", "") or ""),
                    "mtime_ns": int(data.get("mtime_ns", 0) or 0),
                    "workspace_version": current_version,
                }
        elif tool_name == "workspace.file.list" and ok:
            base_path = self._normalize_path(data.get("base_path") or arguments.get("path") or ".")
            key = f"{base_path}|{int(bool(arguments.get('recursive', False)))}"
            lists[key] = {
                "semantic_signature": semantic_signature,
                "snapshot_id": str(data.get("snapshot_id", "") or ""),
                "snapshot_version": int(data.get("snapshot_version", 0) or 0),
                "workspace_version": current_version,
            }

        if ok and tool_name in _MUTATION_TOOLS and bool(execution_flag.get("state_change_committed", False)):
            current_version = self._bump_workspace_version(state)
            snapshot["last_mutation"] = {
                "tool_name": tool_name,
                "workspace_version": current_version,
                "paths": self._extract_call_paths(arguments, data),
            }

        snapshot["reads"] = reads
        snapshot["lists"] = lists
        snapshot["workspace_version"] = self._workspace_version(state)
        state.metadata["workspace_snapshot"] = snapshot

    @staticmethod
    def _is_path_related(a: str, b: str) -> bool:
        left = (a or "").strip().replace("\\", "/").strip("/")
        right = (b or "").strip().replace("\\", "/").strip("/")
        if not left or not right:
            return False
        left_lower = left.lower()
        right_lower = right.lower()
        if left_lower == right_lower:
            return True
        if left_lower.startswith(right_lower + "/") or right_lower.startswith(left_lower + "/"):
            return True
        # Absolute path vs relative path compatibility:
        # `d:/repo/a/b.py` should match `a/b.py`.
        if left_lower.endswith("/" + right_lower) or right_lower.endswith("/" + left_lower):
            return True
        # File-name-only targets should match by basename.
        left_base = left_lower.split("/")[-1]
        right_base = right_lower.split("/")[-1]
        return bool(left_base and right_base and left_base == right_base)

    def _non_mutation_call_related_to_goal(
        self,
        *,
        call_paths: list[str],
        target_paths: list[str],
    ) -> bool:
        if not call_paths or not target_paths:
            return False
        for cp in call_paths:
            for tp in target_paths:
                if self._is_path_related(cp, tp):
                    return True
        return False

    def _evaluate_goal_alignment(
        self,
        state: EngineState,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        ok: bool,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        contract = state.metadata.get("goal_contract", {})
        if not isinstance(contract, dict):
            contract = {}
        target_paths = [self._normalize_path(p) for p in contract.get("target_paths", []) if str(p).strip()]
        has_explicit_target = bool(contract.get("has_explicit_target_path", False))
        requires_side_effect = bool(contract.get("requires_side_effect", False))
        call_paths = self._extract_call_paths(arguments, data)

        path_match: bool | None = None
        if target_paths:
            path_match = any(
                self._is_path_related(call_path, target_path)
                for call_path in call_paths
                for target_path in target_paths
            )

        if tool_name in _VERIFICATION_TOOLS and ok and target_paths:
            verified_paths = set(state.metadata.get("goal_verified_paths", []))
            for path in call_paths:
                for target in target_paths:
                    if self._is_path_related(path, target):
                        verified_paths.add(target)
            if tool_name == "workspace.file.list":
                entries = data.get("entries", [])
                if isinstance(entries, list):
                    for row in entries:
                        if isinstance(row, dict):
                            listed = self._normalize_path(row.get("path"))
                            for target in target_paths:
                                if self._is_path_related(listed, target):
                                    verified_paths.add(target)
            state.metadata["goal_verified_paths"] = sorted(verified_paths)

        tool_can_mutate = tool_name in _MUTATION_TOOLS
        aligned = bool(ok)
        if tool_can_mutate and has_explicit_target and path_match is False:
            aligned = False
        if tool_can_mutate and requires_side_effect and has_explicit_target and path_match is None:
            aligned = False

        return {
            "requires_side_effect": requires_side_effect,
            "target_paths": target_paths,
            "call_paths": call_paths,
            "path_match": path_match,
            "tool_can_mutate": tool_can_mutate,
            "aligned": aligned,
        }

    def _update_goal_contract_from_model(self, state: EngineState, payload: dict[str, Any]) -> None:
        contract = state.metadata.get("goal_contract", {})
        if not isinstance(contract, dict):
            contract = self._derive_goal_contract(state)

        model_contract = payload.get("goal_contract", {})
        if isinstance(model_contract, dict):
            intent_kind = str(model_contract.get("intent_kind", "")).strip().lower()
            if intent_kind in {"state_change", "read_only", "analysis"}:
                # Never downgrade to read_only/analysis after mutation proof has been required.
                if bool(contract.get("requires_side_effect", False)) and intent_kind in {"read_only", "analysis"}:
                    pass
                else:
                    contract["intent_kind"] = intent_kind
            if bool(model_contract.get("requires_side_effect", False)):
                contract["requires_side_effect"] = True
            if bool(model_contract.get("requires_read_proof", False)):
                contract["requires_read_proof"] = True
            model_paths = model_contract.get("target_paths", [])
            if isinstance(model_paths, list):
                merged = [self._normalize_path(p) for p in contract.get("target_paths", []) if str(p).strip()]
                for raw in model_paths:
                    normalized = self._normalize_path(raw)
                    if normalized and normalized not in merged:
                        merged.append(normalized)
                contract["target_paths"] = merged[:6]
                contract["has_explicit_target_path"] = bool(contract["target_paths"])

        completion = payload.get("completion", {})
        if isinstance(completion, dict) and "needs_side_effect_proof" in completion:
            if bool(completion.get("needs_side_effect_proof", False)):
                contract["requires_side_effect"] = True
                contract["source"] = "model_completion"

        calls = payload.get("calls", [])
        if isinstance(calls, list) and any(
            isinstance(item, dict) and str(item.get("tool_name", "")).strip() in _MUTATION_TOOLS for item in calls
        ):
            contract["requires_side_effect"] = True
            contract["source"] = "model_tool_plan"
            if contract.get("intent_kind", "unknown") == "unknown":
                contract["intent_kind"] = "state_change"

        # If model explicitly marks non-state-change intent, allow it to clear side-effect requirement.
        if contract.get("intent_kind") in {"read_only", "analysis"}:
            should_clear = (
                not bool(completion.get("needs_side_effect_proof", False)) if isinstance(completion, dict) else True
            )
            # Never downgrade once side-effect requirement has been raised.
            if should_clear and not bool(contract.get("requires_side_effect", False)):
                contract["requires_side_effect"] = False
                contract["source"] = "model_intent_kind"

        state.metadata["goal_contract"] = contract

    def _build_mainloop_prompt(
        self,
        state: EngineState,
        *,
        tools: list[dict[str, Any]],
        turn_index: int,
        phase: str = "decide",
    ) -> str:
        phase = str(phase or "decide").strip().lower()
        prompt_stage = {
            "action": "mainloop_action",
            "answer": "mainloop_answer",
        }.get(phase, "mainloop_decide")
        if self.prompt_runtime is not None and hasattr(self.prompt_runtime, "build_prompt"):
            prompt_state = state.to_dict()
            prompt_context = {
                "tools": tools,
                "turn_index": turn_index,
                "recent_tool_results": state.metadata.get("recent_tool_results", []),
                "last_parse_error": state.metadata.get("last_parse_error", ""),
                "execution_state": state.metadata.get("execution_state", {}),
                "last_model_plan": state.metadata.get("last_model_plan", {}),
                "goal_contract": state.metadata.get("goal_contract", {}),
                "completion_gate_feedback": state.metadata.get("completion_gate_feedback", {}),
                "context_compaction": state.metadata.get("context_compaction", {}),
                "workspace_snapshot": state.metadata.get("workspace_snapshot", {}),
                "workspace_snapshot_version": state.metadata.get("workspace_snapshot_version", 0),
            }
            return str(
                self.prompt_runtime.build_prompt(
                    prompt_state,
                    prompt_stage,
                    context=prompt_context,
                )
            )

        recent_tool_results = state.metadata.get("recent_tool_results", [])
        if not isinstance(recent_tool_results, list):
            recent_tool_results = []

        return (
            "You are the main execution brain of Self AI.\n"
            "Your job is to solve the task by deciding either final answer or tool calls.\n"
            "Return JSON only. No prose.\n\n"
            "Output schema:\n"
            "{\n"
            '  "type": "final_answer" | "tool_calls",\n'
            '  "response": "final answer when type=final_answer",\n'
            '  "calls": [\n'
            '    {"tool_name":"...", "arguments": {...}, "reason":"..."}\n'
            "  ],\n"
            '  "execution_mode": "optional, short mode label",\n'
            '  "workflow_decision": {"optional": "light decision summary"},\n'
            '  "completion": {\n'
            '    "outcome": "completed|blocked|failed",\n'
            '    "needs_side_effect_proof": true|false,\n'
            '    "needs_read_proof": true|false,\n'
            '    "is_complete": true|false,\n'
            '    "evidence_note": "short note",\n'
            '    "evidence_refs": ["recent evidence_ref ids"],\n'
            '    "remaining_blockers": ["brief blocker descriptions"]\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- Prefer the minimal number of tool calls needed.\n"
            "- Return final_answer directly for explanation, discussion, planning, confirmation, or conversational/session preference updates.\n"
            "- Return tool_calls first for external actions or verification such as workspace writes, reads, tests, storage queries, or project inspection.\n"
            "- Once you call tools, final_answer must be grounded in recent tool results.\n"
            "- If task needs file change, use workspace.file.write/edit.\n"
            "- If file path is uncertain, call workspace.file.list first.\n"
            "- If enough info is available, return final_answer directly.\n"
            "- Use EXACT tool argument keys from tool input_schema.\n"
            "- After a successful tool call that satisfies the task, return final_answer.\n"
            "- Read recent_tool_results.execution_flag first; these flags are source-of-truth of execution.\n"
            "- If goal_contract.target_paths is non-empty, treat it as primary path anchor and avoid introducing unrelated file paths.\n"
            "- Set completion.outcome=completed only when tool evidence proves the requested work succeeded.\n"
            "- Set completion.outcome=blocked or failed when tool evidence proves the work cannot currently be completed; explain exact failed evidence and blockers.\n"
            "- For final_answer with read/state-change requirements, provide completion.evidence_refs.\n"
            "- If completion_gate_feedback.missing is non-empty, satisfy missing proof first.\n"
            "- For non-quick tasks, do not finalize without successful tool execution evidence.\n"
            "- Never output markdown; strict JSON only.\n"
            "- Max 3 tool calls in one turn.\n\n"
            f"Run ID: {state.run_id}\n"
            f"Session ID: {state.session_id}\n"
            f"Turn: {turn_index}\n"
            f"Task: {state.normalized_task or state.task}\n"
            f"Available tools: {self._safe_json(tools, max_len=4000)}\n"
            f"Selected agents so far: {self._safe_json(state.selected_agents, max_len=600)}\n"
            f"Known errors so far: {self._safe_json(state.errors[-5:], max_len=1200)}\n"
            f"Recent tool results: {self._safe_json(recent_tool_results[-5:], max_len=2000)}\n"
            f"Goal contract: {self._safe_json(state.metadata.get('goal_contract', {}), max_len=900)}\n"
            f"Completion gate feedback: {self._safe_json(state.metadata.get('completion_gate_feedback', {}), max_len=900)}\n"
        )

    async def _repair_json_once(
        self,
        state: EngineState,
        *,
        raw_response: str,
        parse_error: str,
        turn_index: int,
        run_context: Any | None = None,
    ) -> dict[str, Any]:
        repair_prompt = (
            "Convert the following text into ONE valid JSON object.\n"
            "Return JSON only. No markdown.\n"
            "Target schema keys: type, response, calls, execution_mode, workflow_decision, plan, goal_contract, completion.\n"
            "type must be final_answer or tool_calls.\n"
            f"Parse error: {parse_error}\n"
            f"Turn: {turn_index}\n"
            "Raw model output:\n"
            f"{raw_response[:8000]}"
        )
        repair_result = await self._call_tool(
            state,
            tool_name="model.generate",
            arguments={
                "prompt": repair_prompt,
                "stage": "mainloop_json_repair",
                "hints": {
                    "goal": "json_repair",
                    "prefer_structured_json": True,
                    "turn_index": turn_index,
                },
            },
            stage=f"mainloop_json_repair_{turn_index}",
            run_context=run_context,
        )
        if not repair_result.ok:
            raise RuntimeError("model.generate failed in json repair")
        repaired_data = repair_result.data if isinstance(repair_result.data, dict) else {}
        repaired_text = str(repaired_data.get("response", "") or "")
        return self._extract_json_object(repaired_text)

    @staticmethod
    def _normalize_call_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
        calls = payload.get("calls", [])
        if isinstance(payload.get("call"), dict):
            calls = [payload["call"]]
        if isinstance(calls, dict):
            calls = [calls]
        if not isinstance(calls, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in calls:
            if not isinstance(item, dict):
                continue
            name = str(item.get("tool_name", "")).strip()
            args = item.get("arguments", {})
            if not name:
                continue
            if not isinstance(args, dict):
                args = {}
            normalized.append(
                {
                    "tool_name": name,
                    "arguments": args,
                    "reason": str(item.get("reason", ""))[:300],
                }
            )
        return normalized[:3]

    def _refresh_execution_state(self, state: EngineState) -> None:
        recent = self._iter_recent_tool_results(state)
        success_calls = sum(1 for item in recent if bool(item.get("ok", False)))
        failed_calls = sum(1 for item in recent if not bool(item.get("ok", False)))
        side_effect_success = 0
        verified_reads = 0
        last_success_tool = ""
        last_failed_tool = ""
        blockers: list[str] = []

        for item in recent:
            flag = item.get("execution_flag", {})
            if isinstance(flag, dict) and bool(flag.get("state_change_committed", False)):
                side_effect_success += 1
            if isinstance(flag, dict) and bool(flag.get("verified_read", False)):
                verified_reads += 1
            if bool(item.get("ok", False)):
                last_success_tool = str(item.get("tool_name", ""))
            else:
                last_failed_tool = str(item.get("tool_name", ""))
                err = item.get("error", {})
                if isinstance(err, dict):
                    err_type = str(err.get("type", "")).strip()
                    if err_type:
                        blockers.append(err_type)

        blocked_count = int(state.metadata.get("completion_gate_blocked_count", 0) or 0)
        control_events = state.metadata.get("control_events", [])
        if not isinstance(control_events, list):
            control_events = []
        contract = state.metadata.get("goal_contract", {})
        if not isinstance(contract, dict):
            contract = {}
        target_paths = contract.get("target_paths", [])
        if not isinstance(target_paths, list):
            target_paths = []
        verified_paths = state.metadata.get("goal_verified_paths", [])
        if not isinstance(verified_paths, list):
            verified_paths = []
        progress = "idle"
        if success_calls > 0:
            progress = "progressing"
        if failed_calls > success_calls:
            progress = "blocked"
        if blocked_count > 0 and side_effect_success == 0:
            progress = "blocked"
        goal_progress = "none"
        if verified_reads > 0:
            goal_progress = "read_verified"
        if side_effect_success > 0:
            goal_progress = "state_changed"
        if side_effect_success > 0 and verified_reads > 0:
            goal_progress = "read_and_state_changed"

        state.metadata["execution_state"] = {
            "progress": progress,
            "goal_progress": goal_progress,
            "successful_calls": success_calls,
            "failed_calls": failed_calls,
            "side_effect_successful_calls": side_effect_success,
            "verified_read_calls": verified_reads,
            "last_success_tool": last_success_tool,
            "last_failed_tool": last_failed_tool,
            "completion_gate_blocked_count": blocked_count,
            "control_event_count": len(control_events),
            "terminal_error_count": len(state.errors),
            "blockers": blockers[-6:],
            "goal_requires_side_effect": bool(contract.get("requires_side_effect", False)),
            "goal_target_path_count": len(target_paths),
            "goal_verified_path_count": len(verified_paths),
            "workspace_snapshot_version": int(state.metadata.get("workspace_snapshot_version", 0) or 0),
        }

    def _note_model_plan(self, state: EngineState, payload: dict[str, Any], *, turn_index: int) -> None:
        plan = payload.get("plan", {})
        plan_obj = plan if isinstance(plan, dict) else {}
        calls = payload.get("calls", [])
        call_count = len(calls) if isinstance(calls, list) else 0
        summary = {
            "turn": turn_index,
            "type": str(payload.get("type", "")).strip().lower(),
            "goal": str(plan_obj.get("goal", "")).strip()[:200],
            "success_criteria": str(plan_obj.get("success_criteria", "")).strip()[:220],
            "next_step": str(plan_obj.get("next_step", "")).strip()[:220],
            "call_count": call_count,
        }
        state.metadata["last_model_plan"] = summary

    def _record_tool_result(
        self,
        state: EngineState,
        *,
        call_signature: str,
        tool_name: str,
        arguments: dict[str, Any],
        ok: bool,
        data: dict[str, Any],
        error: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        recent = state.metadata.setdefault("recent_tool_results", [])
        if not isinstance(recent, list):
            recent = []
            state.metadata["recent_tool_results"] = recent
        goal_check = self._evaluate_goal_alignment(
            state,
            tool_name=tool_name,
            arguments=arguments,
            ok=ok,
            data=data,
        )
        execution_flag = (
            metadata.get("execution_flag", {})
            if isinstance(metadata, dict) and isinstance(metadata.get("execution_flag", {}), dict)
            else {}
        )
        if self._is_verified_read_evidence(
            tool_name=tool_name,
            ok=ok,
            execution_flag=execution_flag,
            goal_check=goal_check,
            data=data,
        ):
            execution_flag = dict(execution_flag)
            execution_flag["verified_read"] = True
        semantic_signature = self._semantic_signature(
            tool_name=tool_name,
            arguments=arguments,
            data=data,
        )
        seq = int(state.metadata.get("tool_result_seq", 0) or 0) + 1
        state.metadata["tool_result_seq"] = seq
        evidence_ref = f"ev:{seq}:{tool_name}"
        item = {
            "evidence_ref": evidence_ref,
            "call_signature": call_signature,
            "semantic_signature": semantic_signature,
            "tool_name": tool_name,
            "ok": bool(ok),
            "data_preview": self._safe_json(data, max_len=700),
            "error": error or {},
            "arguments_preview": self._safe_json(arguments, max_len=400),
            "execution_flag": execution_flag,
            "goal_check": goal_check,
            "workspace_snapshot_version": int(state.metadata.get("workspace_snapshot_version", 0) or 0),
            "snapshot_id": str(data.get("snapshot_id", "") or ""),
        }
        recent.append(item)
        if len(recent) > 20:
            del recent[:-20]
        self._update_workspace_snapshot(
            state,
            tool_name=tool_name,
            ok=ok,
            data=data,
            arguments=arguments,
            execution_flag=execution_flag,
            semantic_signature=semantic_signature,
        )

    @staticmethod
    def _map_block_reason_to_missing(reason: str) -> list[str]:
        normalized = str(reason or "").strip().lower()
        mapping = {
            "execution_evidence_missing": "need_successful_tool_execution",
            "read_evidence_missing": "need_verified_read_evidence",
            "goal_alignment_missing": "need_goal_aligned_state_change",
            "side_effect_evidence_missing": "need_state_change_committed_evidence",
            "model_self_reported_incomplete": "model_reported_remaining_blockers",
            "target_path_not_reported": "need_target_path_mentioned_in_response",
            "evidence_refs_missing": "need_completion_evidence_refs",
            "evidence_refs_invalid": "need_valid_completion_evidence_refs",
            "evidence_refs_missing_side_effect": "need_side_effect_evidence_ref",
            "evidence_refs_missing_read": "need_read_evidence_ref",
            "failed_evidence_missing": "need_failed_tool_evidence_for_blocked_or_failed_outcome",
            "quick_answer_disallowed_by_goal_contract": "quick_answer_not_allowed_for_this_goal",
        }
        return [mapping.get(normalized, normalized or "unknown_completion_gap")]

    def _is_verified_read_evidence(
        self,
        *,
        tool_name: str,
        ok: bool,
        execution_flag: dict[str, Any] | None,
        goal_check: dict[str, Any] | None,
        data: dict[str, Any] | None,
    ) -> bool:
        if not bool(ok):
            return False
        if isinstance(goal_check, dict) and not bool(goal_check.get("aligned", True)):
            return False
        flag = execution_flag if isinstance(execution_flag, dict) else {}
        if bool(flag.get("verified_read", False)):
            return True
        if bool(flag) and not bool(flag.get("executed", True)):
            return False
        permission = str(flag.get("permission", "") or "").strip().lower()
        if permission in _READ_EVIDENCE_PERMISSIONS:
            return True
        payload = data if isinstance(data, dict) else {}
        if isinstance(payload.get("evidence"), list):
            return True
        return False

    def _set_completion_gate_feedback(
        self,
        state: EngineState,
        *,
        reason: str,
        blocked_count: int,
        gate: str,
    ) -> None:
        contract = state.metadata.get("goal_contract", {})
        if not isinstance(contract, dict):
            contract = {}
        target_paths = contract.get("target_paths", [])
        if not isinstance(target_paths, list):
            target_paths = []
        next_hint = {
            "read_evidence_missing": "call workspace.file.read/list or storage.semantic.search to gather verifiable read evidence",
            "side_effect_evidence_missing": "call workspace.file.write/edit/rename/delete to commit required state change",
            "execution_evidence_missing": "execute at least one relevant tool call before final answer",
            "failed_evidence_missing": "cite the failed executed tool result when reporting a blocked or failed outcome",
            "goal_alignment_missing": "use goal_contract.target_paths as the exact mutation targets",
            "model_self_reported_incomplete": "follow remaining_blockers and issue relevant tool calls before final answer",
            "quick_answer_disallowed_by_goal_contract": "switch from quick answer to tool action path first",
        }.get(str(reason or "").strip().lower(), "satisfy missing proof with relevant tool calls")
        state.metadata["completion_gate_feedback"] = {
            "gate": gate,
            "reason": reason,
            "blocked_count": int(blocked_count),
            "missing": self._map_block_reason_to_missing(reason),
            "target_paths": [str(x) for x in target_paths[:6]],
            "next_hint": next_hint,
        }

    def _build_completion_gate_terminal_response(self, state: EngineState) -> str:
        feedback = state.metadata.get("completion_gate_feedback", {})
        if not isinstance(feedback, dict):
            feedback = {}
        missing = feedback.get("missing", [])
        if not isinstance(missing, list):
            missing = []
        missing_text = ", ".join(str(x) for x in missing if str(x).strip()) or "unknown"
        reason = str(feedback.get("reason", "completion_evidence_missing")).strip()
        return (
            "Execution not completed with verifiable evidence. "
            f"Missing: {missing_text}. "
            f"Reason: {reason}. "
            "Please continue with required tool actions and evidence refs."
        )

    def _recent_tool_results_by_ref(self, state: EngineState) -> dict[str, dict[str, Any]]:
        ref_map: dict[str, dict[str, Any]] = {}
        for item in self._iter_recent_tool_results(state):
            ref = str(item.get("evidence_ref", "")).strip()
            if ref:
                ref_map[ref] = item
        return ref_map

    @staticmethod
    def _normalize_completion_evidence_refs(completion_obj: dict[str, Any]) -> list[str]:
        raw = completion_obj.get("evidence_refs", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        refs: list[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text and text not in refs:
                refs.append(text)
        return refs[:8]

    def _call_signature(self, tool_name: str, arguments: dict[str, Any]) -> str:
        payload = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)
        return f"{tool_name}:{payload}"

    def _is_repeated_failed_call(
        self,
        state: EngineState,
        *,
        call_signature: str,
        semantic_signature: str,
        tool_name: str,
        threshold: int = 2,
    ) -> bool:
        recent = state.metadata.get("recent_tool_results", [])
        if not isinstance(recent, list) or not recent:
            return False
        current_ws_version = int(state.metadata.get("workspace_snapshot_version", 0) or 0)
        matched = 0
        for item in reversed(recent[-_VERIFICATION_DUP_WINDOW:]):
            if not isinstance(item, dict):
                continue
            if tool_name in _VERIFICATION_TOOLS:
                if str(item.get("semantic_signature", "")) != semantic_signature:
                    continue
                raw_ws_version = item.get("workspace_snapshot_version", None)
                try:
                    item_ws_version = int(raw_ws_version) if raw_ws_version is not None else -1
                except Exception:
                    item_ws_version = -1
                if item_ws_version != current_ws_version:
                    continue
            else:
                if str(item.get("call_signature", "")) != call_signature:
                    continue
            if bool(item.get("ok", False)):
                return False
            matched += 1
            if matched >= threshold:
                return True
        return False

    def _iter_recent_tool_results(self, state: EngineState) -> list[dict[str, Any]]:
        recent = state.metadata.get("recent_tool_results", [])
        if not isinstance(recent, list):
            return []
        return [item for item in recent if isinstance(item, dict)]

    def _has_successful_tool_call(self, state: EngineState) -> bool:
        for item in reversed(self._iter_recent_tool_results(state)):
            flag = item.get("execution_flag", {})
            if isinstance(flag, dict) and bool(flag.get("ok", False)):
                return True
            if not flag and bool(item.get("ok", False)):
                return True
        return False

    def _has_failed_tool_evidence(self, state: EngineState) -> bool:
        for item in reversed(self._iter_recent_tool_results(state)):
            if bool(item.get("ok", False)):
                continue
            flag = item.get("execution_flag", {})
            if isinstance(flag, dict):
                # Loop guards may record non-executed pseudo-results; those
                # are useful feedback but not external evidence of a blocker.
                if bool(flag.get("executed", False)):
                    return True
                continue
            return True
        return False

    def _has_entered_tool_loop(self, state: EngineState) -> bool:
        return bool(self._iter_recent_tool_results(state))

    def _has_successful_side_effect_tool_call(self, state: EngineState) -> bool:
        for item in reversed(self._iter_recent_tool_results(state)):
            flag = item.get("execution_flag", {})
            if not isinstance(flag, dict):
                continue
            goal_check = item.get("goal_check", {})
            if isinstance(goal_check, dict) and not bool(goal_check.get("aligned", True)):
                continue
            if bool(flag.get("state_change_committed", False)):
                return True
        return False

    def _has_verified_read_evidence(self, state: EngineState) -> bool:
        for item in reversed(self._iter_recent_tool_results(state)):
            flag = item.get("execution_flag", {})
            if isinstance(flag, dict) and bool(flag.get("verified_read", False)):
                return True
        metadata = state.metadata if isinstance(state.metadata, dict) else {}
        goal_contract = metadata.get("goal_contract", {})
        if not isinstance(goal_contract, dict):
            goal_contract = {}
        target_paths = goal_contract.get("target_paths", [])
        if not isinstance(target_paths, list):
            target_paths = []
        # Treat injected session-memory context as read proof only for
        # in-session recall questions without explicit external targets.
        if not target_paths and bool(goal_contract.get("requires_read_proof", False)):
            stats = metadata.get("chat_recent_turns_stats", {})
            if isinstance(stats, dict):
                kept = int(stats.get("kept_turns", 0) or 0)
                if kept > 0:
                    return True
        return False

    def _is_repeated_success_call(
        self,
        state: EngineState,
        *,
        call_signature: str,
        semantic_signature: str,
        tool_name: str,
    ) -> bool:
        recent = state.metadata.get("recent_tool_results", [])
        if not isinstance(recent, list) or not recent:
            return False
        current_ws_version = int(state.metadata.get("workspace_snapshot_version", 0) or 0)
        for item in reversed(recent[-_VERIFICATION_DUP_WINDOW:]):
            if not isinstance(item, dict):
                continue
            if not bool(item.get("ok", False)):
                continue
            if tool_name in _VERIFICATION_TOOLS:
                if str(item.get("semantic_signature", "")) != semantic_signature:
                    continue
                raw_ws_version = item.get("workspace_snapshot_version", None)
                try:
                    item_ws_version = int(raw_ws_version) if raw_ws_version is not None else -1
                except Exception:
                    item_ws_version = -1
                if item_ws_version == current_ws_version:
                    return True
                continue
            if str(item.get("call_signature", "")) == call_signature:
                return True
        return False

    def _has_goal_aligned_side_effect(self, state: EngineState) -> bool:
        contract = state.metadata.get("goal_contract", {})
        if not isinstance(contract, dict):
            return False
        if not bool(contract.get("requires_side_effect", False)):
            return True
        for item in reversed(self._iter_recent_tool_results(state)):
            if not bool(item.get("ok", False)):
                continue
            goal_check = item.get("goal_check", {})
            if not isinstance(goal_check, dict):
                continue
            if not bool(goal_check.get("tool_can_mutate", False)):
                continue
            if bool(goal_check.get("aligned", False)):
                return True
        return False

    def _build_auto_final_response(self, state: EngineState) -> str:
        contract = state.metadata.get("goal_contract", {})
        if not isinstance(contract, dict):
            contract = {}
        target_paths = contract.get("target_paths", [])
        if not isinstance(target_paths, list):
            target_paths = []
        if target_paths:
            return f"Done. Verified execution evidence exists for target path(s): {', '.join(str(x) for x in target_paths[:3])}."
        return "Done. Verified execution evidence exists for the requested action."

    def _should_block_non_mutation_call(self, state: EngineState, tool_name: str) -> tuple[bool, str]:
        contract = state.metadata.get("goal_contract", {})
        if not isinstance(contract, dict):
            return False, ""
        if not bool(contract.get("requires_side_effect", False)):
            return False, ""
        if not bool(contract.get("has_explicit_target_path", False)):
            return False, ""
        if self._has_goal_aligned_side_effect(state):
            return False, ""
        if tool_name in _MUTATION_TOOLS:
            return False, ""
        target_paths = [self._normalize_path(p) for p in contract.get("target_paths", []) if str(p).strip()]
        call_paths: list[str] = []
        pending_call_arguments = state.metadata.get("pending_call_arguments", {})
        if isinstance(pending_call_arguments, dict):
            call_paths = self._extract_call_paths(pending_call_arguments, {})
        if self._non_mutation_call_related_to_goal(call_paths=call_paths, target_paths=target_paths):
            return False, ""
        return True, "goal_requires_mutation_first"

    def _final_answer_can_commit(self, state: EngineState, payload: dict[str, Any]) -> tuple[bool, str]:
        raw_decision = payload.get("workflow_decision", {})
        decision = raw_decision if isinstance(raw_decision, dict) else {}
        if not decision and isinstance(state.workflow_decision, dict):
            decision = state.workflow_decision

        use_quick_answer = bool(decision.get("use_quick_answer", False))
        requires_context = bool(decision.get("requires_context", not use_quick_answer))
        completion = payload.get("completion", {})
        completion_obj = completion if isinstance(completion, dict) else {}
        raw_outcome = str(completion_obj.get("outcome", "")).strip().lower()
        outcome = raw_outcome if raw_outcome in {"completed", "blocked", "failed"} else "completed"
        needs_side_effect_proof = bool(completion_obj.get("needs_side_effect_proof", False))
        goal_contract = state.metadata.get("goal_contract", {})
        if not isinstance(goal_contract, dict):
            goal_contract = {}
        if bool(goal_contract.get("requires_side_effect", False)):
            needs_side_effect_proof = True
        # Read-proof gate is system-driven (goal contract), not model-self-declared.
        # Only explicit requires_read_proof should enable this gate.
        needs_read_proof = bool(goal_contract.get("requires_read_proof", False))
        if completion_obj and outcome == "completed" and not bool(completion_obj.get("is_complete", True)):
            return False, "model_self_reported_incomplete"

        requires_external_proof = (
            needs_side_effect_proof
            or needs_read_proof
            or bool(goal_contract.get("requires_side_effect", False))
            or bool(goal_contract.get("requires_read_proof", False))
        )

        # Mainloop owns the evidence decision:
        # - final_answer before any task tool call means the model judged no
        #   external evidence is required.
        # - once the model chooses tool_calls, final_answer must close the
        #   tool-result evidence loop.
        # - explicit completion/goal proof requirements still force evidence.
        entered_tool_loop = self._has_entered_tool_loop(state)
        if not entered_tool_loop and not requires_external_proof:
            return True, ""

        if entered_tool_loop and outcome in {"blocked", "failed"}:
            if self._has_failed_tool_evidence(state):
                return True, ""
            return False, "failed_evidence_missing"

        if (entered_tool_loop or requires_external_proof) and not self._has_successful_tool_call(state):
            return False, "execution_evidence_missing"

        if needs_read_proof and not self._has_verified_read_evidence(state):
            return False, "read_evidence_missing"
        if bool(goal_contract.get("requires_side_effect", False)) and not self._has_goal_aligned_side_effect(state):
            return False, "goal_alignment_missing"
        if needs_side_effect_proof and not self._has_successful_side_effect_tool_call(state):
            return False, "side_effect_evidence_missing"
        return True, ""

    async def _execute_model_turn(
        self,
        state: EngineState,
        *,
        turn_index: int,
        phase: str = "decide",
        expected_type: str | None = None,
        run_context: Any | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        phase = str(phase or "decide").strip().lower()
        tools = self._tool_catalog(run_context=run_context)
        prompt = self._build_mainloop_prompt(state, tools=tools, turn_index=turn_index, phase=phase)
        if self.prompt_runtime is not None and hasattr(self.prompt_runtime, "consume_last_build_stats"):
            stats = self.prompt_runtime.consume_last_build_stats()
            if isinstance(stats, dict) and stats:
                state.metadata["prompt_stats"] = stats
                await self._trace(
                    state,
                    "prompt.mainloop.stats",
                    stage="mainloop",
                    status="built",
                    fields=stats,
                    run_context=run_context,
                )
        result = await self._call_tool(
            state,
            tool_name="model.generate",
            arguments={
                "prompt": prompt,
                "stage": "mainloop_decide",
                "hints": {
                    "goal": "main_loop_decision_json",
                    "prefer_structured_json": True,
                    "turn_index": turn_index,
                    "decision_phase": phase,
                    "enable_thinking": False,
                    "max_tokens": 32768,
                },
            },
            stage=f"mainloop_turn_{turn_index}",
            run_context=run_context,
        )
        if not result.ok:
            raise RuntimeError("model.generate failed in main loop")
        data = result.data if isinstance(result.data, dict) else {}
        model_name = str(data.get("model", "") or "")
        response = str(data.get("response", "") or "")
        try:
            payload = self._extract_json_object(response)
            state.metadata["last_parse_error"] = ""
        except Exception as parse_exc:
            state.metadata["last_parse_error"] = f"{type(parse_exc).__name__}: {parse_exc}"
            try:
                payload = await self._repair_json_once(
                    state,
                    raw_response=response,
                    parse_error=state.metadata["last_parse_error"],
                    turn_index=turn_index,
                    run_context=run_context,
                )
            except Exception:
                raise RuntimeError(f"ModelJSONParseError[{phase}]")
            state.metadata["last_parse_error"] = ""
        payload_type = str(payload.get("type", "")).strip().lower()
        if payload_type not in {"final_answer", "tool_calls"}:
            raise RuntimeError(f"ModelJSONTypeError[{phase}]")
        if expected_type and payload_type != expected_type:
            raise RuntimeError(
                f"ModelPhaseTypeMismatch[{phase}]: expected={expected_type}, got={payload_type}"
            )
        if phase == "decide":
            state.metadata["last_model_payload"] = payload
        return payload_type, payload, model_name

    def _merge_workflow_decision(
        self,
        state: EngineState,
        *,
        decision_type: str,
        payload: dict[str, Any],
        turn_index: int,
    ) -> None:
        raw_decision = payload.get("workflow_decision", {})
        if not isinstance(raw_decision, dict):
            raw_decision = {}

        mode = str(raw_decision.get("execution_mode") or payload.get("execution_mode") or "").strip().lower()
        if not mode:
            mode = "model_driven"
        use_quick = bool(
            raw_decision.get("use_quick_answer")
            if "use_quick_answer" in raw_decision
            else mode == "quick_answer"
        )

        selected_agents = raw_decision.get("selected_agents", state.selected_agents)
        if not isinstance(selected_agents, list):
            selected_agents = list(state.selected_agents)

        decision = {
            "execution_mode": mode,
            "next_node": str(
                raw_decision.get("next_node")
                or ("final_answer" if decision_type == "final_answer" else "tool_calls")
            ),
            "skip_research": bool(raw_decision.get("skip_research", use_quick)),
            "skip_debate": bool(raw_decision.get("skip_debate", True)),
            "use_quick_answer": use_quick,
            "requires_context": bool(raw_decision.get("requires_context", not use_quick)),
            "requires_debate": bool(raw_decision.get("requires_debate", False)),
            "max_iterations": int(raw_decision.get("max_iterations", 1) or 1),
            "selected_agents": [str(x) for x in selected_agents][:8],
            "rationale": str(
                raw_decision.get("rationale")
                or payload.get("reason")
                or "model_driven"
            )[:300],
            "policy_version": "mainloop_v1",
            "metadata": {
                "source": "model_mainloop",
                "turn": turn_index,
            },
        }
        maintenance = payload.get("constraint_maintenance", {})
        if isinstance(maintenance, dict):
            decision["constraint_maintenance"] = {
                "enabled": bool(maintenance.get("enabled", False)),
                "reason": str(maintenance.get("reason", "") or "")[:240],
            }
        state.workflow_decision = decision

    async def run(self, state: EngineState, *, run_context: Any | None = None) -> EngineState:
        state.metadata.setdefault("engine_mode", "mainloop_v1")
        state.metadata.setdefault("agent_runtime_used", False)
        state.metadata.setdefault("revision_count", 0)
        state.metadata.setdefault("max_revision_iterations", 1)
        state.metadata.setdefault("revision_performed", False)
        state.metadata.setdefault("revision_target_agent", None)
        state.metadata.setdefault("pipeline", ["mainloop"])
        state.metadata.setdefault("pipeline_source", "mainloop")
        state.metadata.setdefault("recent_tool_results", [])
        state.metadata.setdefault("last_parse_error", "")
        state.metadata.setdefault("execution_state", {})
        state.metadata.setdefault("last_model_plan", {})
        state.metadata.setdefault("goal_contract", self._derive_goal_contract(state))
        state.metadata.setdefault("goal_verified_paths", [])
        state.metadata.setdefault("no_progress_repeat_count", 0)
        state.metadata.setdefault("workspace_snapshot_version", 0)
        state.metadata.setdefault("workspace_snapshot", {"reads": {}, "lists": {}, "workspace_version": 0})
        self._refresh_execution_state(state)

        await self._trace(
            state,
            "goal_contract.infer.skipped",
            stage="goal_contract_infer",
            status="skipped",
            fields={"reason": "mainloop_decides_tool_or_final_answer"},
            run_context=run_context,
        )
        self._refresh_execution_state(state)

        max_turns = int(state.metadata.get("max_turns", 10) or 10)
        max_turns = max(1, min(max_turns, 30))
        max_failures = int(state.metadata.get("max_failures", 3) or 3)
        max_failures = max(1, min(max_failures, 10))
        max_tool_calls_total = int(state.metadata.get("max_tool_calls_total", 18) or 18)
        max_tool_calls_total = max(3, min(max_tool_calls_total, 60))
        context_token_threshold = int(state.metadata.get("context_token_threshold", 2500) or 2500)
        context_token_threshold = max(800, min(context_token_threshold, 20000))
        tool_calls_total = 0
        failure_count = 0

        await self._trace(
            state,
            "engine.run.start",
            stage="engine",
            status="start",
            fields={"mode": "mainloop_v1", "max_turns": max_turns},
            run_context=run_context,
        )
        await self._trace(
            state,
            "engine.pipeline.selected",
            stage="engine",
            status="selected",
            fields={
                "run_id": state.run_id,
                "session_id": state.session_id,
                "execution_mode": "mainloop_v1",
                "pipeline_name": "mainloop",
                "stages": ["mainloop"],
                "stage_count": 1,
                "pipeline_source": "mainloop",
                "max_tool_calls_total": max_tool_calls_total,
            },
            run_context=run_context,
        )
        await self.state_store.save_state(state, run_context=run_context)

        for turn in range(1, max_turns + 1):
            self._microcompact_recent_results(state)
            est_tokens = self._estimate_context_tokens(state)
            if est_tokens > context_token_threshold:
                self._auto_compact_recent_results(state)
                await self._trace(
                    state,
                    "context.compact.auto",
                    stage="mainloop",
                    status="compacted",
                    fields={
                        "turn": turn,
                        "estimated_tokens_before": est_tokens,
                        "threshold": context_token_threshold,
                        "recent_tool_results_count": len(state.metadata.get("recent_tool_results", [])),
                    },
                    run_context=run_context,
                )
            else:
                await self._trace(
                    state,
                    "context.compact.micro",
                    stage="mainloop",
                    status="ok",
                    fields={
                        "turn": turn,
                        "estimated_tokens": est_tokens,
                        "threshold": context_token_threshold,
                    },
                    run_context=run_context,
                )
            await self._trace(
                state,
                "loop.turn.start",
                stage="mainloop",
                status="start",
                fields={"turn": turn},
                run_context=run_context,
            )
            try:
                decision_type, payload, model_name = await self._execute_model_turn(
                    state,
                    turn_index=turn,
                    phase="decide",
                    expected_type=None,
                    run_context=run_context,
                )
                if model_name:
                    state.model = model_name
                self._merge_workflow_decision(
                    state,
                    decision_type=decision_type,
                    payload=payload,
                    turn_index=turn,
                )
                self._note_model_plan(state, payload, turn_index=turn)
                self._update_goal_contract_from_model(state, payload)
            except Exception as exc:
                failure_count += 1
                state.add_error(
                    type(exc).__name__,
                    str(exc),
                    stage="mainloop_decide",
                )
                await self._trace(
                    state,
                    "loop.turn.error",
                    stage="mainloop",
                    status="error",
                    fields={"turn": turn, "error": type(exc).__name__},
                    run_context=run_context,
                )
                await self._trace(
                    state,
                    "engine.stage.error",
                    stage="mainloop",
                    status="error",
                    fields={"turn": turn, "error": type(exc).__name__},
                    run_context=run_context,
                )
                if failure_count >= max_failures:
                    break
                continue

            failure_count = 0
            if decision_type == "final_answer":
                candidate_response = str(payload.get("response", "") or "").strip()
                if not candidate_response:
                    candidate_response = "No response generated."
                can_commit, block_reason = self._final_answer_can_commit(state, payload)
                if not can_commit:
                    blocked_count = int(state.metadata.get("completion_gate_blocked_count", 0) or 0) + 1
                    state.metadata["completion_gate_blocked_count"] = blocked_count
                    self._set_completion_gate_feedback(
                        state,
                        reason=block_reason,
                        blocked_count=blocked_count,
                        gate="completion_gate",
                    )
                    self._refresh_execution_state(state)
                    state.add_control_event(
                        "CompletionGateBlocked",
                        stage="mainloop",
                        reason=block_reason,
                        metadata={"turn": turn},
                    )
                    self._refresh_execution_state(state)
                    await self._trace(
                        state,
                        "mainloop.final_answer.blocked",
                        stage="mainloop",
                        status="blocked",
                        fields={
                            "turn": turn,
                            "reason": block_reason,
                            "blocked_count": blocked_count,
                        },
                        run_context=run_context,
                    )
                    await self._save_stage_output(
                        state,
                        f"mainloop_turn_{turn}_blocked",
                        {
                            "turn": turn,
                            "reason": block_reason,
                            "blocked_count": blocked_count,
                            "response_preview": candidate_response[:220],
                            "execution_state": state.metadata.get("execution_state", {}),
                            "last_model_plan": state.metadata.get("last_model_plan", {}),
                            "completion_gate_feedback": state.metadata.get("completion_gate_feedback", {}),
                        },
                        run_context=run_context,
                    )
                    if blocked_count >= 3:
                        state.response = self._build_completion_gate_terminal_response(state)
                        state.metadata["completion_gate_terminal"] = True
                        state.add_error(
                            "CompletionNotVerifiedTerminal",
                            f"completion gate terminal block: {block_reason}",
                            stage="mainloop",
                            metadata={"reason": block_reason, "blocked_count": blocked_count},
                        )
                        self._refresh_execution_state(state)
                        await self._trace(
                            state,
                            "mainloop.final_answer.terminal_block",
                            stage="mainloop",
                            status="failed",
                            fields={"turn": turn, "blocked_count": blocked_count, "reason": block_reason},
                            run_context=run_context,
                        )
                        break
                    continue
                decision_raw = payload.get("workflow_decision", {})
                decision_obj = decision_raw if isinstance(decision_raw, dict) else {}
                if not decision_obj and isinstance(state.workflow_decision, dict):
                    decision_obj = state.workflow_decision

                state.response = candidate_response
                await self._save_stage_output(
                    state,
                    "mainloop_final_answer",
                    {
                        "turn": turn,
                        "response_preview": state.response[:240],
                        "execution_state": state.metadata.get("execution_state", {}),
                        "last_model_plan": state.metadata.get("last_model_plan", {}),
                    },
                    run_context=run_context,
                )
                await self._trace(
                    state,
                    "loop.turn.end",
                    stage="mainloop",
                    status="final_answer",
                    fields={"turn": turn},
                    run_context=run_context,
                )
                break

            calls = self._normalize_call_list(payload)
            if not calls:
                state.add_error(
                    "MainLoopNoCalls",
                    "tool_calls decision returned empty calls",
                    stage="mainloop",
                )
                await self._trace(
                    state,
                    "loop.turn.error",
                    stage="mainloop",
                    status="error",
                    fields={"turn": turn, "error": "empty_calls"},
                    run_context=run_context,
                )
                await self._trace(
                    state,
                    "engine.stage.error",
                    stage="mainloop",
                    status="error",
                    fields={"turn": turn, "error": "empty_calls"},
                    run_context=run_context,
                )
                continue

            for call in calls:
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_signature = self._call_signature(tool_name, arguments)
                semantic_signature = self._semantic_signature(
                    tool_name=tool_name,
                    arguments=arguments,
                    data={},
                )
                state.metadata["pending_call_arguments"] = arguments
                should_block, block_reason = self._should_block_non_mutation_call(state, tool_name)
                if should_block:
                    state.add_error(
                        "GoalActionMismatch",
                        f"blocked non-mutation tool before required mutation: {tool_name}",
                        stage="mainloop",
                        metadata={"reason": block_reason, "tool_name": tool_name},
                    )
                    self._record_tool_result(
                        state,
                        call_signature=call_signature,
                        tool_name=tool_name,
                        arguments=arguments,
                        ok=False,
                        data={},
                        error={"type": "GoalActionMismatch", "message": block_reason},
                        metadata={"execution_flag": {"called": True, "executed": False, "ok": False}},
                    )
                    self._refresh_execution_state(state)
                    await self._trace(
                        state,
                        "loop.tool.blocked_goal_mismatch",
                        stage="mainloop",
                        status="blocked",
                        fields={"turn": turn, "tool": tool_name, "reason": block_reason},
                        run_context=run_context,
                    )
                    continue
                if tool_name == "model.generate":
                    state.add_error(
                        "MainLoopInvalidTool",
                        "model.generate is not callable from tool_calls",
                        stage="mainloop",
                    )
                    continue
                if self._is_repeated_failed_call(
                    state,
                    call_signature=call_signature,
                    semantic_signature=semantic_signature,
                    tool_name=tool_name,
                ):
                    state.add_error(
                        "RepeatedToolCallBlocked",
                        f"blocked repeated failed tool call: {tool_name}",
                        stage="mainloop",
                        metadata={"call_signature": call_signature},
                    )
                    self._record_tool_result(
                        state,
                        call_signature=call_signature,
                        tool_name=tool_name,
                        arguments=arguments,
                        ok=False,
                        data={},
                        error={"type": "RepeatedToolCallBlocked", "message": "blocked by loop guard"},
                        metadata={"execution_flag": {"called": True, "executed": False, "ok": False}},
                    )
                    self._refresh_execution_state(state)
                    await self._trace(
                        state,
                        "loop.tool.blocked_repetition",
                        stage="mainloop",
                        status="blocked",
                        fields={"turn": turn, "tool": tool_name},
                        run_context=run_context,
                    )
                    continue
                if self._is_repeated_success_call(
                    state,
                    call_signature=call_signature,
                    semantic_signature=semantic_signature,
                    tool_name=tool_name,
                ):
                    repeat_count = int(state.metadata.get("no_progress_repeat_count", 0) or 0) + 1
                    state.metadata["no_progress_repeat_count"] = repeat_count
                    state.add_error(
                        "RepeatedToolCallNoProgress",
                        f"blocked repeated successful tool call without progress: {tool_name}",
                        stage="mainloop",
                        metadata={"call_signature": call_signature, "repeat_count": repeat_count},
                    )
                    self._record_tool_result(
                        state,
                        call_signature=call_signature,
                        tool_name=tool_name,
                        arguments=arguments,
                        ok=False,
                        data={},
                        error={"type": "RepeatedToolCallNoProgress", "message": "blocked by loop convergence guard"},
                        metadata={"execution_flag": {"called": True, "executed": False, "ok": False}},
                    )
                    self._refresh_execution_state(state)
                    await self._trace(
                        state,
                        "loop.tool.blocked_repeated_success",
                        stage="mainloop",
                        status="blocked",
                        fields={"turn": turn, "tool": tool_name, "repeat_count": repeat_count},
                        run_context=run_context,
                    )
                    continue

                tool_calls_total += 1
                if tool_calls_total > max_tool_calls_total:
                    state.add_error(
                        "MainLoopToolBudgetExceeded",
                        "tool call budget exceeded; finalize with current context",
                        stage="mainloop",
                    )
                    await self._trace(
                        state,
                        "loop.tool.budget_exceeded",
                        stage="mainloop",
                        status="error",
                        fields={"turn": turn, "tool_calls_total": tool_calls_total},
                        run_context=run_context,
                    )
                    break

                result = await self._call_tool(
                    state,
                    tool_name=tool_name,
                    arguments=arguments,
                    stage=f"mainloop_turn_{turn}",
                    run_context=run_context,
                )
                if tool_name == "agent.run":
                    agent_name = str(arguments.get("agent_name", "")).strip()
                    if agent_name and agent_name not in state.selected_agents:
                        state.selected_agents.append(agent_name)
                        state.metadata["agent_runtime_used"] = True
                self._record_tool_result(
                    state,
                    call_signature=call_signature,
                    tool_name=tool_name,
                    arguments=arguments,
                    ok=result.ok,
                    data=result.data if isinstance(result.data, dict) else {},
                    error=result.error,
                    metadata=result.metadata if isinstance(result.metadata, dict) else {},
                )
                state.metadata["no_progress_repeat_count"] = 0
                state.metadata["pending_call_arguments"] = {}
                self._refresh_execution_state(state)
                if tool_name == "review.quality_gate.decide":
                    gate = state.quality_gate if isinstance(state.quality_gate, dict) else {}
                    decision = str(gate.get("decision", "")).lower()
                    if decision in {"pass", "warn"}:
                        state.metadata["revision_count"] = int(gate.get("current_iteration", 0) or 0)
                    if decision in {"revise", "fail"}:
                        state.metadata["revision_performed"] = True
                        state.metadata["revision_target_agent"] = gate.get("revision_target_agent")

            if tool_calls_total > max_tool_calls_total:
                break

            await self._save_stage_output(
                state,
                f"mainloop_turn_{turn}",
                {
                    "turn": turn,
                    "call_count": len(calls),
                    "selected_agents": list(state.selected_agents),
                    "error_count": len(state.errors),
                    "execution_state": state.metadata.get("execution_state", {}),
                    "last_model_plan": state.metadata.get("last_model_plan", {}),
                },
                run_context=run_context,
            )
            await self._trace(
                state,
                "loop.turn.end",
                stage="mainloop",
                status="tool_calls_done",
                fields={"turn": turn, "call_count": len(calls)},
                run_context=run_context,
            )
            if (
                int(state.metadata.get("no_progress_repeat_count", 0) or 0) >= 2
                and self._has_goal_aligned_side_effect(state)
                and not state.response
            ):
                state.response = self._build_auto_final_response(state)
                state.metadata["auto_finalized_due_to_no_progress"] = True
                await self._trace(
                    state,
                    "mainloop.auto_finalize",
                    stage="mainloop",
                    status="completed",
                    fields={"turn": turn, "reason": "repeated_success_without_progress"},
                    run_context=run_context,
                )
                break

        if not state.response:
            contract = state.metadata.get("goal_contract", {})
            if (
                isinstance(contract, dict)
                and bool(contract.get("requires_side_effect", False))
                and not self._has_goal_aligned_side_effect(state)
            ):
                state.response = (
                    "Execution is not verified: required state-change action was not completed with "
                    "goal-aligned tool evidence."
                )
            if int(state.metadata.get("completion_gate_blocked_count", 0) or 0) > 0:
                state.response = (
                    "Execution is not verified: the model produced verbal completion without "
                    "sufficient successful tool-execution evidence. Ask it to complete tool "
                    "actions first, then return the final answer."
                )
                if not any(
                    isinstance(err, dict) and str(err.get("type", "")) == "CompletionNotVerifiedTerminal"
                    for err in state.errors
                ):
                    feedback = state.metadata.get("completion_gate_feedback", {})
                    reason = (
                        str(feedback.get("reason", "completion_evidence_missing") or "completion_evidence_missing")
                        if isinstance(feedback, dict)
                        else "completion_evidence_missing"
                    )
                    state.add_error(
                        "CompletionNotVerifiedTerminal",
                        f"completion gate ended without verified completion: {reason}",
                        stage="mainloop",
                        metadata={
                            "reason": reason,
                            "blocked_count": int(state.metadata.get("completion_gate_blocked_count", 0) or 0),
                        },
                    )
                    self._refresh_execution_state(state)
            synth = (
                state.agent_outputs.get("synthesizer", {})
                if isinstance(state.agent_outputs.get("synthesizer"), dict)
                else {}
            )
            synth_output = str(synth.get("output", "") or "").strip()
            state.response = state.response or synth_output or "No response generated."
            if not state.model:
                meta = synth.get("metadata", {}) if isinstance(synth.get("metadata"), dict) else {}
                state.model = str(meta.get("model", "") or state.model)

        await self._save_stage_output(
            state,
            "mainloop_finalize",
            {
                "has_response": bool(state.response),
                "error_count": len(state.errors),
                "selected_agents": list(state.selected_agents),
                "execution_state": state.metadata.get("execution_state", {}),
                "last_model_plan": state.metadata.get("last_model_plan", {}),
            },
            run_context=run_context,
        )
        await self._trace(
            state,
            "engine.run.end",
            stage="engine",
            status="completed",
            fields={"mode": "mainloop_v1", "errors": len(state.errors)},
            run_context=run_context,
        )
        await self.state_store.save_state(state, run_context=run_context)
        return state
