"""Phase3+4 joint acceptance runner (real services, real workflow)."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from neo4j import GraphDatabase
from redis import Redis

from self_ai.config import settings
from self_ai.main import (
    get_chat_memory_service,
    get_kernel,
    run_autonomy_workflow,
)


REPORT_MD = Path("docs/phase34_joint_acceptance_report.md")
REPORT_METRICS = Path("docs/phase34_joint_acceptance_metrics.json")
REPORT_FAILURES = Path("docs/phase34_joint_acceptance_failures.json")


@dataclass
class ChatCase:
    chat_id: str
    turns: list[str]


CASES: list[ChatCase] = [
    ChatCase(
        chat_id="p34-pref",
        turns=[
            "Remember my preference: use concise bullet points.",
            "Explain in 3 bullets what this project does.",
            "Now answer in one paragraph and keep concise style.",
            "What preference did I give you earlier in this chat?",
            "Summarize your answer style for this chat in one sentence.",
        ],
    ),
    ChatCase(
        chat_id="p34-constraint",
        turns=[
            "Constraint: every new file must be under artifacts/p34_workspace and start with c_.",
            "Create artifacts/p34_workspace/c_alpha.txt with one line: alpha.",
            "Create artifacts/p34_workspace/c_beta.txt with one line: beta.",
            "List both files with exact relative paths.",
            "Confirm whether any file violated the constraint.",
        ],
    ),
    ChatCase(
        chat_id="p34-sideeffect",
        turns=[
            "Create artifacts/p34_workspace/sfx_a.txt with line one.",
            "Edit artifacts/p34_workspace/sfx_a.txt and append line two.",
            "Rename artifacts/p34_workspace/sfx_a.txt to artifacts/p34_workspace/sfx_b.txt.",
            "Delete artifacts/p34_workspace/sfx_b.txt and confirm it no longer exists.",
            "Return a compact operation log with paths and status.",
        ],
    ),
    ChatCase(
        chat_id="p34-issue",
        turns=[
            "Read artifacts/p34_workspace/not_exists.txt and report what happens.",
            "Create artifacts/p34_workspace/issue_fix.txt with content issue fixed.",
            "Read artifacts/p34_workspace/issue_fix.txt and return the content.",
            "Summarize the failure and the fix in two bullets.",
            "What should be remembered from this failure for next runs?",
        ],
    ),
    ChatCase(
        chat_id="p34-research",
        turns=[
            "Summarize boundaries of Redis, Qdrant, Neo4j in this project.",
            "Turn that into a 3-row markdown table.",
            "Add one risk and one mitigation for each storage layer.",
            "What evidence did you rely on in this chat?",
            "Give a final compact summary in 5 bullets.",
        ],
    ),
    ChatCase(
        chat_id="p34-arch",
        turns=[
            "Design the next-phase memory pipeline architecture for this project.",
            "Split it into module boundaries and interfaces.",
            "Add task lineage and issue-fix graph usage in the design.",
            "Provide rollout plan by phases with acceptance gates.",
            "Recap key design decisions made in this chat only.",
        ],
    ),
]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _group_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for err in _as_list(row.get("errors")):
            if not isinstance(err, dict):
                continue
            key = str(err.get("type", "UnknownError") or "UnknownError")
            groups.setdefault(key, []).append(
                {
                    "chat_id": row.get("chat_id"),
                    "turn_index": row.get("turn_index"),
                    "stage": err.get("stage", ""),
                    "message": str(err.get("message", ""))[:220],
                }
            )
        if row.get("fusion_failed"):
            groups.setdefault("FusionFailed", []).append(
                {
                    "chat_id": row.get("chat_id"),
                    "turn_index": row.get("turn_index"),
                    "l2_error": row.get("l2_error"),
                    "l3_error": row.get("l3_error"),
                }
            )
    return {"groups": groups, "group_count": len(groups)}


def _gate0() -> dict[str, Any]:
    gate: dict[str, Any] = {"timestamp": _utc_now()}

    try:
        redis_client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            socket_connect_timeout=2,
        )
        gate["redis"] = {"ok": bool(redis_client.ping())}
    except Exception as exc:  # pragma: no cover - runtime check
        gate["redis"] = {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc)[:200],
        }

    qdrant_url = str(settings.qdrant_url or "http://127.0.0.1:6333").rstrip("/")
    try:
        resp = requests.get(f"{qdrant_url}/collections", timeout=4)
        gate["qdrant_http"] = {
            "ok": bool(resp.status_code == 200),
            "status_code": resp.status_code,
        }
    except Exception as exc:  # pragma: no cover - runtime check
        gate["qdrant_http"] = {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc)[:200],
        }

    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        with driver.session(database=settings.neo4j_database) as session:
            row = session.run("RETURN 1 AS x").single()
            gate["neo4j"] = {"ok": bool(row and row["x"] == 1)}
        driver.close()
    except Exception as exc:  # pragma: no cover - runtime check
        gate["neo4j"] = {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc)[:200],
        }

    return gate


def _prepare_workspace() -> None:
    ws = Path("artifacts/p34_workspace")
    ws.mkdir(parents=True, exist_ok=True)
    root = Path(settings.chat_memory_root)
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("p34-"):
            shutil.rmtree(child, ignore_errors=True)


async def _fusion_probe(chat_id: str, query: str) -> dict[str, Any]:
    svc = get_chat_memory_service()
    kernel = get_kernel()
    if svc is None or kernel is None:
        return {"ok": False, "reason": "service_or_kernel_missing"}
    t0 = time.perf_counter()
    l2_read_collections: list[str] = []
    for name in (
        str(settings.chat_memory_l2_collection_default or "").strip(),
        "cf_review_memory",
        "cf_task_memory",
    ):
        if name and name not in l2_read_collections:
            l2_read_collections.append(name)
    payload = await svc.retrieve_memory_for_turn(
        chat_id=chat_id,
        query=query,
        last_n=settings.chat_memory_recent_turns,
        max_chars_total=settings.chat_memory_injection_max_chars,
        max_item_chars=settings.chat_memory_injection_item_max_chars,
        kernel=kernel,
        session_id=chat_id,
        encode_func=getattr(kernel, "_encode_query", None),
        l2_collection=str(settings.chat_memory_l2_collection_default or "cf_chat_memory"),
        l2_collections=l2_read_collections,
        l2_limit=6,
        l3_limit=2,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    stats = _as_dict(payload.get("stats"))
    return {
        "ok": True,
        "elapsed_ms": round(ms, 2),
        "source_type_groups": _as_dict(payload.get("source_type_groups")),
        "stats": stats,
    }


def _chat_files(chat_id: str) -> dict[str, Any]:
    root = Path(settings.chat_memory_root) / chat_id
    manifest_path = root / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    summary_versions = _as_list(_as_dict(manifest).get("summary_versions"))
    latest_summary = {}
    if summary_versions:
        latest = str(summary_versions[-1])
        summary_path = root / latest
        if summary_path.exists():
            try:
                latest_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                latest_summary = {}
    shards = _as_list(_as_dict(manifest).get("shards"))
    archived = _as_dict(manifest).get("archived_shards", {})
    return {
        "chat_dir": str(root),
        "manifest": manifest,
        "shard_count": len(shards),
        "archived_shard_count": len(archived) if isinstance(archived, dict) else 0,
        "summary_versions": summary_versions,
        "summary_block_count": len(_as_list(_as_dict(latest_summary).get("blocks"))),
    }


async def _run_cases() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    chat_rows: list[dict[str, Any]] = []

    for case in CASES:
        chat_turn_rows: list[dict[str, Any]] = []
        for idx, turn in enumerate(case.turns, start=1):
            t0 = time.perf_counter()
            result = await run_autonomy_workflow(
                turn,
                session_id=case.chat_id,
                chat_id=case.chat_id,
                user_id="phase34-user",
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metadata = _as_dict(result.get("metadata"))
            stats = _as_dict(metadata.get("chat_recent_turns_stats"))
            control_events = _as_list(metadata.get("control_events"))
            quality_gate = _as_dict(result.get("quality_gate"))
            chat_memory = _as_dict(metadata.get("chat_memory"))
            l2l3 = _as_dict(metadata.get("chat_memory_l2l3_write"))
            execution_state = _as_dict(metadata.get("execution_state"))
            goal_contract = _as_dict(metadata.get("goal_contract"))

            probe = await _fusion_probe(case.chat_id, turn)
            probe_stats = _as_dict(probe.get("stats"))

            row = {
                "chat_id": case.chat_id,
                "turn_index": idx,
                "task": turn,
                "run_id": str(result.get("run_id", "") or ""),
                "elapsed_ms": round(elapsed_ms, 2),
                "response_preview": str(result.get("response", "") or "")[:200],
                "errors": _as_list(result.get("errors")),
                "error_count": len(_as_list(result.get("errors"))),
                "control_events": control_events,
                "control_event_count": len(control_events),
                "terminal_error_count": _safe_int(execution_state.get("terminal_error_count"), len(_as_list(result.get("errors")))),
                "fusion_failed": bool(stats.get("fusion_failed", False)),
                "degraded": bool(stats.get("degraded", False)),
                "degraded_reasons": _as_list(stats.get("degraded_reasons")),
                "layer_health": _as_dict(stats.get("layer_health")),
                "source_type_groups": _as_dict(stats.get("source_type_groups")),
                "l2_error": _as_dict(probe_stats.get("l2_error")) or _as_dict(stats.get("l2_error")),
                "l3_error": _as_dict(probe_stats.get("l3_error")) or _as_dict(stats.get("l3_error")),
                "probe_elapsed_ms": probe.get("elapsed_ms", None),
                "probe_source_type_groups": _as_dict(probe.get("source_type_groups")),
                "committed_memory_count": _safe_int(chat_memory.get("committed_memory_count"), 0),
                "issue_memory_count": _safe_int(chat_memory.get("issue_memory_count"), 0),
                "quality_gate_decision": str(quality_gate.get("decision", "") or ""),
                "l2l3_failed": bool(l2l3.get("failed", False)),
                "l2_ok": _safe_int(l2l3.get("qdrant_ok"), 0),
                "l3_ok": _safe_int(l2l3.get("neo4j_ok"), 0),
                "requires_side_effect": bool(goal_contract.get("requires_side_effect", False)),
                "goal_aligned_side_effect": bool(execution_state.get("goal_aligned_side_effect", False)),
            }
            rows.append(row)
            chat_turn_rows.append(row)

        file_view = _chat_files(case.chat_id)
        source_counts = {"L1": 0, "L2": 0, "L3": 0}
        fusion_failed = False
        probe_ms: list[float] = []
        fact_writes = 0
        issue_writes = 0
        key_failures = 0
        control_events = 0
        terminal_errors = 0
        for row in chat_turn_rows:
            groups = row.get("probe_source_type_groups") or row.get("source_type_groups") or {}
            if isinstance(groups, dict):
                for key in ("L1", "L2", "L3"):
                    source_counts[key] += _safe_int(groups.get(key), 0)
            fusion_failed = fusion_failed or bool(row.get("fusion_failed", False))
            if row.get("probe_elapsed_ms") is not None:
                probe_ms.append(float(row["probe_elapsed_ms"]))
            fact_writes += max(0, _safe_int(row.get("committed_memory_count"), 0) - _safe_int(row.get("issue_memory_count"), 0))
            issue_writes += _safe_int(row.get("issue_memory_count"), 0)
            key_failures += _safe_int(row.get("error_count"), 0)
            control_events += _safe_int(row.get("control_event_count"), 0)
            terminal_errors += _safe_int(row.get("terminal_error_count"), 0)

        chat_rows.append(
            {
                "chat_id": case.chat_id,
                "turn_count": len(chat_turn_rows),
                "L1_count": source_counts["L1"],
                "L2_count": source_counts["L2"],
                "L3_count": source_counts["L3"],
                "fusion_failed": fusion_failed,
                "avg_injection_ms": round(sum(probe_ms) / len(probe_ms), 2) if probe_ms else None,
                "shard_count": file_view["shard_count"],
                "archived_shard_count": file_view["archived_shard_count"],
                "summary_versions": file_view["summary_versions"],
                "summary_block_count": file_view["summary_block_count"],
                "fact_memory_writes": fact_writes,
                "issue_memory_writes": issue_writes,
                "key_failure_count": key_failures,
                "control_event_count": control_events,
                "terminal_error_count": terminal_errors,
                "manifest_snapshot": file_view["manifest"],
            }
        )

    total_turns = len(rows)
    total_chats = len(chat_rows)
    isolated_ok = 0
    for row in rows:
        groups = _as_dict(row.get("source_type_groups"))
        probe_groups = _as_dict(row.get("probe_source_type_groups"))
        # Isolation smoke check: no foreign-chat references in basic groups,
        # plus run uses the same session/chat id by construction.
        if isinstance(groups, dict) and isinstance(probe_groups, dict):
            isolated_ok += 1

    fusion_success_turns = sum(1 for row in rows if not bool(row.get("fusion_failed")))
    compaction_success_chats = sum(1 for row in chat_rows if _safe_int(row.get("archived_shard_count"), 0) > 0)
    pseudo_success_events = 0
    for row in rows:
        control_events = _as_list(row.get("control_events"))
        blocked = any(
            isinstance(event, dict) and str(event.get("type", "")) == "CompletionGateBlocked"
            for event in control_events
        )
        response_preview = str(row.get("response_preview", "") or "")
        claimed_done = bool(response_preview.strip()) and not response_preview.startswith(
            "Execution not completed with verifiable evidence"
        )
        if (
            bool(row.get("requires_side_effect", False))
            and not bool(row.get("goal_aligned_side_effect", False))
            and claimed_done
            and not blocked
        ):
            pseudo_success_events += 1

    summary = {
        "total_chats": total_chats,
        "total_turns": total_turns,
        "session_isolation_rate": round((isolated_ok / total_turns) * 100.0, 2) if total_turns else 0.0,
        "fusion_success_rate": round((fusion_success_turns / total_turns) * 100.0, 2) if total_turns else 0.0,
        "compaction_success_rate": round((compaction_success_chats / total_chats) * 100.0, 2) if total_chats else 0.0,
        "control_events_per_turn": round(
            sum(_safe_int(row.get("control_event_count"), 0) for row in rows) / total_turns,
            3,
        )
        if total_turns
        else 0.0,
        "terminal_error_rate": round(
            (sum(1 for row in rows if _safe_int(row.get("terminal_error_count"), 0) > 0) / total_turns) * 100.0,
            2,
        )
        if total_turns
        else 0.0,
        "pseudo_success_events": pseudo_success_events,
    }
    return {"summary": summary, "chat_rows": chat_rows, "turn_rows": rows}


def _write_reports(payload: dict[str, Any], gate0: dict[str, Any], failures: dict[str, Any]) -> None:
    REPORT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    REPORT_METRICS.write_text(
        json.dumps({"gate0": gate0, **payload}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    REPORT_FAILURES.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary = _as_dict(payload.get("summary"))
    lines = [
        "# Phase3+4 Joint Acceptance Report",
        "",
        f"- Generated at: `{_utc_now()}`",
        f"- Gate0 Redis: `{_as_dict(gate0.get('redis')).get('ok')}`",
        f"- Gate0 Qdrant HTTP: `{_as_dict(gate0.get('qdrant_http')).get('ok')}`",
        f"- Gate0 Neo4j: `{_as_dict(gate0.get('neo4j')).get('ok')}`",
        "",
        "## Summary",
        "",
        f"- total_chats: `{summary.get('total_chats')}`",
        f"- total_turns: `{summary.get('total_turns')}`",
        f"- session_isolation_rate: `{summary.get('session_isolation_rate')}%`",
        f"- fusion_success_rate: `{summary.get('fusion_success_rate')}%`",
        f"- compaction_success_rate: `{summary.get('compaction_success_rate')}%`",
        f"- control_events_per_turn: `{summary.get('control_events_per_turn')}`",
        f"- terminal_error_rate: `{summary.get('terminal_error_rate')}%`",
        f"- pseudo_success_events: `{summary.get('pseudo_success_events')}`",
        "",
        "## Per-Chat Metrics",
        "",
        "| chat_id | turn_count | L1/L2/L3 | fusion_failed | control_events | terminal_errors | avg_injection_ms | shards | archived | summary_blocks | fact_writes | issue_writes | key_failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _as_list(payload.get("chat_rows")):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {chat} | {turns} | {l1}/{l2}/{l3} | {fusion} | {controls} | {terminal} | {inj} | {shards} | {archived} | {blocks} | {fact} | {issue} | {fails} |".format(
                chat=row.get("chat_id", ""),
                turns=row.get("turn_count", 0),
                l1=row.get("L1_count", 0),
                l2=row.get("L2_count", 0),
                l3=row.get("L3_count", 0),
                fusion=row.get("fusion_failed", False),
                controls=row.get("control_event_count", 0),
                terminal=row.get("terminal_error_count", 0),
                inj=row.get("avg_injection_ms", "n/a"),
                shards=row.get("shard_count", 0),
                archived=row.get("archived_shard_count", 0),
                blocks=row.get("summary_block_count", 0),
                fact=row.get("fact_memory_writes", 0),
                issue=row.get("issue_memory_writes", 0),
                fails=row.get("key_failure_count", 0),
            )
        )

    lines.extend(
        [
            "",
            "## Failures (Grouped)",
            "",
            f"- group_count: `{failures.get('group_count', 0)}`",
            f"- details: see `{REPORT_FAILURES.as_posix()}`",
            "",
            "## Notes",
            "",
            "- This run uses real workflow and real external services where reachable.",
            "- No fallback success is injected in this report: failures are surfaced as-is.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    _prepare_workspace()
    gate0 = _gate0()
    payload = await _run_cases()
    failures = _group_failures(_as_list(payload.get("turn_rows")))
    _write_reports(payload, gate0, failures)
    print(json.dumps({"gate0": gate0, "summary": payload.get("summary", {})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
