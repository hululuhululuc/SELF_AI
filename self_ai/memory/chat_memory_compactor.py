# coding=utf-8
"""Phase-4 chat memory shard compactor (summary + archive index)."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..text_utils import shorten_text
from .chat_memory_store import ChatMemoryStore


def _summary_preview(lines: list[dict[str, Any]], *, max_chars: int) -> str:
    chunks: list[str] = []
    for row in lines:
        if not isinstance(row, dict):
            continue
        text = str(row.get("retrieval_summary", "") or "").strip()
        if text:
            chunks.append(text)
        if len(chunks) >= 8:
            break
    return shorten_text(" | ".join(chunks), max_chars=max_chars)


def _read_shard_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    except Exception:
        return []
    return rows


def compact_chat(
    *,
    store: ChatMemoryStore,
    chat_id: str,
    min_shards: int = 3,
    summary_max_chars: int = 1200,
    max_docs: int = 200,
) -> dict[str, Any]:
    """Compact old shards into a summary file and manifest archive index."""
    candidates = store.list_compaction_candidates(
        chat_id=chat_id,
        min_shards=min_shards,
        max_candidates=max_docs,
    )
    if not candidates:
        return {
            "chat_id": chat_id,
            "compacted": False,
            "reason": "no_candidates",
            "candidate_count": 0,
        }

    manifest = store.read_manifest(chat_id=chat_id)
    safe_chat_id = str(manifest.get("chat_id", chat_id) or chat_id)
    chat_dir = (store.root_dir / safe_chat_id).resolve()

    blocks: list[dict[str, Any]] = []
    compacted_files: list[str] = []
    last_turn_id = int(manifest.get("last_compacted_turn_id", 0) or 0)
    for shard in candidates:
        if not isinstance(shard, dict):
            continue
        file_name = str(shard.get("file", "") or "").strip()
        if not file_name:
            continue
        shard_path = chat_dir / file_name
        if not shard_path.exists():
            continue
        rows = _read_shard_rows(shard_path)
        if not rows:
            continue
        turn_ids = [int(r.get("turn_id", 0) or 0) for r in rows if int(r.get("turn_id", 0) or 0) > 0]
        first_turn = min(turn_ids) if turn_ids else None
        last_turn = max(turn_ids) if turn_ids else None
        if last_turn is not None:
            last_turn_id = max(last_turn_id, last_turn)
        preview = _summary_preview(rows, max_chars=max(120, int(summary_max_chars)))
        digest = sha256(
            ("\n".join(str(r.get("content_hash", "") or "") for r in rows)).encode("utf-8", errors="ignore")
        ).hexdigest()
        block_id = f"{safe_chat_id}:{file_name}:{first_turn or 0}:{last_turn or 0}"
        blocks.append(
            {
                "block_id": block_id,
                "shard_file": file_name,
                "turn_range": [first_turn, last_turn],
                "turn_count": len(turn_ids),
                "preview": preview,
                "content_hash": f"sha256:{digest}",
                "refs": {
                    "run_ids": list(
                        {
                            str(r.get("run_id", "") or "")
                            for r in rows
                            if str(r.get("run_id", "") or "")
                        }
                    )[:20],
                },
            }
        )
        compacted_files.append(file_name)

    if not blocks:
        return {
            "chat_id": safe_chat_id,
            "compacted": False,
            "reason": "no_blocks",
            "candidate_count": len(candidates),
        }

    next_version = len(manifest.get("summary_versions", []) if isinstance(manifest.get("summary_versions"), list) else []) + 1
    summary_file = f"summary-v{next_version}.json"
    summary_payload = {
        "chat_id": safe_chat_id,
        "summary_file": summary_file,
        "version": next_version,
        "block_count": len(blocks),
        "blocks": blocks,
    }
    (chat_dir / summary_file).write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    archive_report = store.mark_shards_archived(
        chat_id=chat_id,
        shard_files=compacted_files,
        summary_file=summary_file,
        summary_block_ids=[str(b.get("block_id", "")) for b in blocks],
        last_compacted_turn_id=last_turn_id,
    )
    return {
        "chat_id": safe_chat_id,
        "compacted": True,
        "summary_file": summary_file,
        "block_count": len(blocks),
        "archived_count": len(compacted_files),
        "last_compacted_turn_id": last_turn_id,
        "archive": archive_report,
    }

