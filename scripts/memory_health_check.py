# coding=utf-8
"""Real L1/L2/L3 memory health check for Self AI.

The script writes a unique synthetic memory turn, verifies local replay, writes
and searches Qdrant with the local BGE-M3 embedder, records a small Neo4j graph,
and saves machine-readable artifacts for release/eval preflight evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient  # noqa: E402

from self_ai.config import resolve_project_path, settings  # noqa: E402
from self_ai.graph.graph_models import GraphEdge, GraphNode  # noqa: E402
from self_ai.memory.chat_memory_service import ChatMemoryService  # noqa: E402
from self_ai.memory.chat_memory_store import ChatMemoryStore  # noqa: E402
from self_ai.memory.neo4j_store import Neo4jGraphMemory  # noqa: E402
from self_ai.memory.qdrant_store import QdrantMemory  # noqa: E402
from self_ai.schemas import EvidenceItem  # noqa: E402
from self_ai.tools import _encode_text  # noqa: E402

RUNS_ROOT = ROOT_DIR / "benchmarks" / "runs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf-8",
    )


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    checks = payload.get("checks", {})
    lines = [
        "# Self AI Memory Health Check",
        "",
        f"Run: `{payload.get('run_id', '')}`",
        f"Generated: `{payload.get('generated_at', '')}`",
        f"Overall OK: `{payload.get('ok', False)}`",
        "",
        "## Layer Results",
        "",
        "| Layer | OK | Evidence |",
        "|---|---:|---|",
    ]
    for key, label in (("l1", "L1 local chat memory"), ("l2", "L2 Qdrant semantic memory"), ("l3", "L3 Neo4j graph memory")):
        row = checks.get(key, {}) if isinstance(checks, dict) else {}
        lines.append(
            "| {label} | `{ok}` | {evidence} |".format(
                label=label,
                ok=bool(row.get("ok", False)),
                evidence=str(row.get("evidence", "") or row.get("error", ""))[:220],
            )
        )
    lines.extend(
        [
            "",
            "## Paths",
            "",
            f"- Embedder path: `{payload.get('embedder_path', '')}`",
            f"- Chat memory root: `{payload.get('chat_memory_root', '')}`",
            f"- Qdrant collection: `{payload.get('qdrant_collection', '')}`",
            f"- Neo4j URI: `{payload.get('neo4j_uri', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(slots=True)
class CheckContext:
    run_id: str
    chat_id: str
    token: str
    output_dir: Path
    qdrant_collection: str


def _result(ok: bool, **fields: Any) -> dict[str, Any]:
    return {"ok": bool(ok), **fields}


def _check_l1(ctx: CheckContext) -> dict[str, Any]:
    root = resolve_project_path(settings.chat_memory_root)
    store = ChatMemoryStore(
        root_dir=root,
        max_shard_bytes=settings.chat_memory_max_shard_bytes,
        max_turns_per_shard=settings.chat_memory_max_turns_per_shard,
    )
    service = ChatMemoryService(store)
    constraint = f"Memory health constraint token {ctx.token}: prefer concise release evidence."
    applied = store.apply_global_constraint_ops(
        chat_id=ctx.chat_id,
        ops=[{"op": "add", "content": constraint}],
        source="memory_health_check",
    )
    result_payload = {
        "response": f"Self AI memory health response with token {ctx.token}.",
        "run_id": ctx.run_id,
        "session_id": ctx.chat_id,
        "metadata": {
            "execution_state": {
                "side_effect_done": False,
                "goal_aligned_side_effect": False,
            },
            "recent_tool_results": [],
        },
        "errors": [],
        "quality_gate": {"decision": "pass"},
    }
    write = store.append_turn(
        chat_id=ctx.chat_id,
        user_id="memory-health",
        run_id=ctx.run_id,
        task=f"Memory health L1 write token {ctx.token}",
        result=result_payload,
        logs=[],
        memory_decision={
            "decision_source": "memory_health_check",
            "profiles": {
                "turn_summary": {
                    "memory_type": "turn_summary",
                    "memory_class": "fact",
                    "target_collection": settings.chat_memory_l2_collection_default,
                    "importance": 0.75,
                    "decay_mode": "slow",
                    "pinned": False,
                    "confidence": 0.95,
                    "rationale": "Synthetic health-check memory should be retrievable.",
                },
                "response_summary": {
                    "memory_type": "response_summary",
                    "memory_class": "fact",
                    "target_collection": settings.chat_memory_l2_collection_default,
                    "importance": 0.75,
                    "decay_mode": "slow",
                    "pinned": False,
                    "confidence": 0.95,
                    "rationale": "Synthetic health-check response should be retrievable.",
                },
            },
        },
    )
    replay = service.replay_turns(chat_id=ctx.chat_id, limit=20, ascending=True)
    injection = service.load_recent_turns_for_injection(
        chat_id=ctx.chat_id,
        last_n=3,
        max_chars_total=4000,
        max_item_chars=900,
    )
    constraints = service.load_global_constraints(chat_id=ctx.chat_id)
    replay_text = json.dumps(replay, ensure_ascii=False, default=str)
    constraints_text = json.dumps(constraints, ensure_ascii=False, default=str)
    ok = (
        bool(write.get("turn_id"))
        and ctx.token in replay_text
        and int(injection.get("kept_turns", 0) or 0) >= 1
        and ctx.token in constraints_text
    )
    return _result(
        ok,
        evidence=f"turn_id={write.get('turn_id')} replay_count={len(replay)} injection={injection.get('kept_turns')} constraints={len(constraints.get('items', []))}",
        write=write,
        replay_count=len(replay),
        injection=injection,
        global_constraints=constraints,
        chat_dir=str(root / ctx.chat_id),
    )


async def _check_l2(ctx: CheckContext) -> dict[str, Any]:
    client = QdrantClient(
        url=settings.qdrant_url,
        check_compatibility=False,
        trust_env=False,
        timeout=max(float(settings.qdrant_timeout_s), 10.0),
    )
    memory = QdrantMemory(
        client=client,
        embedding_model="BAAI/bge-m3",
        strict_schema=bool(settings.qdrant_strict_schema),
    )
    content = (
        f"Self AI L2 memory health token {ctx.token}. "
        "This point proves BGE-M3 embedding, Qdrant upsert, chat scoped metadata, and search readback."
    )
    start = time.perf_counter()
    vector = _encode_text(content)
    encode_ms = int((time.perf_counter() - start) * 1000)
    evidence = EvidenceItem(
        content=content,
        source_type="chat_memory:health_check",
        source=f"chat:{ctx.chat_id}",
        chunk_id=f"{ctx.chat_id}-l2-health",
        score=1.0,
        created_at=_utc_now(),
        embedding_model="BAAI/bge-m3",
        metadata={
            "chat_id": ctx.chat_id,
            "run_id": ctx.run_id,
            "turn_id": 1,
            "memory_type": "health_check",
            "memory_class": "fact",
            "importance": 1.0,
            "confidence": 1.0,
            "decay_mode": "none",
            "pinned": True,
        },
    )
    point_id = str(uuid5(NAMESPACE_URL, f"{ctx.qdrant_collection}:{ctx.chat_id}:l2-health"))
    upsert_ok = await memory.upsert_memory(
        collection_name=ctx.qdrant_collection,
        evidence=evidence,
        dense_vector=vector,
        point_id=point_id,
        wait=True,
    )
    query_vector = _encode_text(f"Find Self AI L2 memory health token {ctx.token}")
    hits = await memory.search(
        collection_name=ctx.qdrant_collection,
        dense_vector=query_vector,
        limit=5,
        min_score=-1.0,
    )
    matching_hits = [
        item.model_dump(mode="json")
        for item in hits
        if ctx.token in str(item.content)
        and (item.metadata or {}).get("chat_id") == ctx.chat_id
    ]
    ok = bool(upsert_ok and matching_hits)
    return _result(
        ok,
        evidence=f"vector_dim={len(vector)} encode_ms={encode_ms} upsert={upsert_ok} hits={len(hits)} scoped_matches={len(matching_hits)}",
        vector_dim=len(vector),
        encode_ms=encode_ms,
        upsert_ok=upsert_ok,
        hit_count=len(hits),
        matching_hits=matching_hits[:3],
    )


def _check_l3(ctx: CheckContext) -> dict[str, Any]:
    if not bool(settings.neo4j_enabled):
        return _result(False, error="NEO4J_ENABLED=false")
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return _result(False, error=f"neo4j import failed: {type(exc).__name__}: {exc}")

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        memory = Neo4jGraphMemory(
            driver=driver,
            database=settings.neo4j_database,
            enabled=True,
        )
        task_id = f"health:task:{ctx.run_id}"
        evidence_id = f"health:evidence:{ctx.run_id}"
        task_ok = memory.upsert_node(
            GraphNode(
                id=task_id,
                label="Task",
                properties={
                    "run_id": ctx.run_id,
                    "chat_id": ctx.chat_id,
                    "token": ctx.token,
                    "created_at": _utc_now(),
                },
            )
        )
        evidence_ok = memory.upsert_node(
            GraphNode(
                id=evidence_id,
                label="Evidence",
                properties={
                    "run_id": ctx.run_id,
                    "chat_id": ctx.chat_id,
                    "token": ctx.token,
                    "summary": "Self AI L3 memory health evidence",
                },
            )
        )
        edge_ok = memory.upsert_edge(
            GraphEdge(
                source_id=task_id,
                target_id=evidence_id,
                relation="USED_EVIDENCE",
                properties={"run_id": ctx.run_id, "token": ctx.token},
            )
        )
        rows = memory._run_read(
            "MATCH (t:Task {id: $task_id})-[r:USED_EVIDENCE]->(e:Evidence {id: $evidence_id}) "
            "RETURN t.id AS task_id, e.id AS evidence_id, r.token AS token LIMIT 1",
            {"task_id": task_id, "evidence_id": evidence_id},
        )
        lineage = memory.find_task_lineage(ctx.run_id, limit=10)
        ok = bool(task_ok and evidence_ok and edge_ok and rows and rows[0].get("token") == ctx.token)
        return _result(
            ok,
            evidence=f"node_task={task_ok} node_evidence={evidence_ok} edge={edge_ok} readback={len(rows)} lineage_edges={len(lineage.edges)}",
            task_ok=task_ok,
            evidence_ok=evidence_ok,
            edge_ok=edge_ok,
            readback=rows,
            lineage=lineage.model_dump(mode="json"),
        )
    finally:
        driver.close()


async def _run(args: argparse.Namespace) -> int:
    run_id = f"memory-health-{_slug_ts()}"
    chat_id = args.chat_id or run_id
    ctx = CheckContext(
        run_id=run_id,
        chat_id=chat_id,
        token=f"mh-{_slug_ts()}",
        output_dir=(RUNS_ROOT / run_id).resolve(),
        qdrant_collection=args.qdrant_collection,
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    embedder_path = resolve_project_path(settings.embedder_path)
    payload: dict[str, Any] = {
        "schema_version": "self_ai.memory_health.v1",
        "run_id": ctx.run_id,
        "chat_id": ctx.chat_id,
        "token": ctx.token,
        "generated_at": _utc_now(),
        "project_root": str(ROOT_DIR),
        "output_dir": str(ctx.output_dir),
        "embedder_path": str(embedder_path),
        "embedder_path_exists": embedder_path.exists(),
        "chat_memory_root": str(resolve_project_path(settings.chat_memory_root)),
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": ctx.qdrant_collection,
        "neo4j_enabled": bool(settings.neo4j_enabled),
        "neo4j_uri": settings.neo4j_uri,
        "settings": {
            "chat_memory_enabled": bool(settings.chat_memory_enabled),
            "chat_memory_fusion_enabled": bool(settings.chat_memory_fusion_enabled),
            "chat_memory_l2_collection_default": settings.chat_memory_l2_collection_default,
            "qdrant_strict_schema": bool(settings.qdrant_strict_schema),
            "graph_read_enabled": bool(settings.graph_read_enabled),
        },
        "checks": {},
    }

    try:
        payload["checks"]["l1"] = _check_l1(ctx)
    except Exception as exc:
        payload["checks"]["l1"] = _result(False, error=f"{type(exc).__name__}: {exc}")

    try:
        payload["checks"]["l2"] = await _check_l2(ctx)
    except Exception as exc:
        payload["checks"]["l2"] = _result(False, error=f"{type(exc).__name__}: {exc}")

    try:
        payload["checks"]["l3"] = _check_l3(ctx)
    except Exception as exc:
        payload["checks"]["l3"] = _result(False, error=f"{type(exc).__name__}: {exc}")

    payload["ok"] = all(bool(row.get("ok", False)) for row in payload["checks"].values())
    payload["finished_at"] = _utc_now()
    _write_json(ctx.output_dir / "memory_health.json", payload)
    _write_report(ctx.output_dir / "report.md", payload)
    print(json.dumps({"ok": payload["ok"], "output_dir": str(ctx.output_dir), "checks": payload["checks"]}, ensure_ascii=False, default=_jsonable))
    return 0 if payload["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real Self AI L1/L2/L3 memory health checks.")
    parser.add_argument("--chat-id", default="", help="Optional chat_id. Defaults to a unique health id.")
    parser.add_argument(
        "--qdrant-collection",
        default="cf_eval_cases",
        help="Qdrant collection used for synthetic health evidence.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
