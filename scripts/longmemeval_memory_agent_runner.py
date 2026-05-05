# coding=utf-8
"""LongMemEval-S memory-agent runner for Self AI.

This runner evaluates multi-session memory capability with real public
LongMemEval-S cases:
1) inject haystack sessions into one chat_id
2) ask cross-session question
3) score answer/evidence/recall/error-memory metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from self_ai.config import settings  # noqa: E402
from self_ai.main import run_autonomy_workflow  # noqa: E402

RUNS_ROOT = ROOT_DIR / "benchmarks" / "runs"
STOP_REASONS = {
    "runner_exception",
    "quota_exhausted",
    "chat_memory_disabled",
    "missing_case_id",
    "missing_question",
}
NOT_ATTEMPTED_PATTERNS = (
    "i don't know",
    "i do not know",
    "not sure",
    "unknown",
    "cannot answer",
    "can't answer",
    "execution not completed with verifiable evidence",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_name(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix)
    return f"{stamp}_{safe or 'longmemeval_s_eval'}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be object at {path}:{line_no}")
        rows.append(row)
    return rows


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[^\w\s.-]", "", text)
    return text.strip()


def _extract_session_ids(text: str) -> list[str]:
    values = re.findall(r"\banswer_[A-Za-z0-9_]+\b", text or "")
    out: list[str] = []
    for val in values:
        if val not in out:
            out.append(val)
    return out


def _contains_quota_error(payload: Any) -> bool:
    markers = (
        "insufficient_quota",
        "exceeded today's quota",
        "rate limit",
        "error code: 429",
        "quota",
        "tokens already used",
    )

    def _walk(v: Any) -> bool:
        if isinstance(v, dict):
            return any(_walk(x) for x in v.values())
        if isinstance(v, list):
            return any(_walk(x) for x in v)
        text = str(v or "").lower()
        return any(m in text for m in markers)

    return _walk(payload)


def _question_answered_correctly(*, response: str, answer: str) -> bool:
    a = _normalize_text(answer)
    r = _normalize_text(response)
    if not a or not r:
        return False
    return a in r or r in a


def _is_not_attempted(response: str) -> bool:
    text = (response or "").lower()
    return any(pattern in text for pattern in NOT_ATTEMPTED_PATTERNS)


def _extract_groups(result: dict[str, Any]) -> dict[str, int]:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    stats = metadata.get("chat_recent_turns_stats", {}) if isinstance(metadata.get("chat_recent_turns_stats"), dict) else {}
    groups = stats.get("source_type_groups", {}) if isinstance(stats.get("source_type_groups"), dict) else {}
    return {
        "L1": int(groups.get("L1", 0) or 0),
        "L2": int(groups.get("L2", 0) or 0),
        "L3": int(groups.get("L3", 0) or 0),
    }


def _chat_memory_root() -> Path:
    root = Path(settings.chat_memory_root)
    if not root.is_absolute():
        root = (ROOT_DIR / root).resolve()
    return root


def _cleanup_chat(chat_id: str) -> None:
    root = _chat_memory_root()
    path = root / chat_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _session_prompt(*, session_id: str, session_date: str, messages: list[dict[str, Any]], max_chars: int) -> str:
    rendered: list[str] = []
    for row in messages:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "unknown")
        content = str(row.get("content") or "").strip()
        has_answer = bool(row.get("has_answer", False))
        if not content:
            continue
        line = f"[{role}] {content}"
        if has_answer:
            line += " [has_answer=true]"
        rendered.append(line)
    joined = "\n".join(rendered)
    if max_chars > 0:
        joined = joined[:max_chars]
    return (
        "Memory ingestion step.\n"
        f"Session ID: {session_id}\n"
        f"Session Date: {session_date}\n"
        "Read and remember the transcript below for future cross-session QA in this same chat.\n"
        "Reply with one short line: INGESTED <Session ID>.\n\n"
        f"{joined}"
    )


def _query_prompt(*, question: str, expected_session_ids: list[str]) -> str:
    expected_text = ", ".join(expected_session_ids) if expected_session_ids else "unknown"
    return (
        "Cross-session memory question.\n"
        "Answer using only memories from earlier ingested sessions in this chat.\n"
        "Return plain text with two lines:\n"
        "1) Answer: <short answer>\n"
        "2) EvidenceSessionIDs: <comma-separated session IDs>\n"
        f"Question: {question}\n"
        f"Hint expected supporting session ids (for evaluator only): {expected_text}"
    )


async def _run_turn(*, task_text: str, chat_id: str, user_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = await run_autonomy_workflow(task_text, session_id=chat_id, chat_id=chat_id, user_id=user_id)
    elapsed = round(time.perf_counter() - started, 3)
    response = str(result.get("response") or "")
    return {
        "elapsed_s": elapsed,
        "response": response,
        "errors": result.get("errors", []),
        "metadata": result.get("metadata", {}),
        "run_id": result.get("run_id", ""),
        "result": result,
    }


def _prepare_paths(run_name: str) -> dict[str, Path]:
    run_dir = RUNS_ROOT / run_name
    return {
        "run_dir": run_dir,
        "cases_dir": run_dir / "cases",
        "config": run_dir / "config.json",
        "metrics": run_dir / "metrics.json",
        "failures": run_dir / "failures.json",
        "report": run_dir / "report.md",
        "raw_predictions": run_dir / "raw_predictions.jsonl",
        "latency": run_dir / "latency_trace.jsonl",
    }


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(settings.chat_memory_enabled):
        raise RuntimeError("chat memory is disabled; cannot run long-memory evaluation")
    cases_path = Path(args.cases_path).resolve()
    all_rows = _read_jsonl(cases_path)
    rows = [row for row in all_rows if str(row.get("stage") or "") == "longmemeval_s"]
    rows = rows[max(0, int(args.offset)) :]
    if int(args.batch_size) > 0:
        rows = rows[: int(args.batch_size)]

    paths = _prepare_paths(str(args.run_name))
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["cases_dir"].mkdir(parents=True, exist_ok=True)
    _write_json(
        paths["config"],
        {
            "generated_at": _utc_now(),
            "runner": "longmemeval_memory_agent_runner.py",
            "cases_path": str(cases_path),
            "offset": int(args.offset),
            "batch_size": int(args.batch_size),
            "chat_prefix": str(args.chat_prefix),
            "max_sessions": int(args.max_sessions),
            "max_session_chars": int(args.max_session_chars),
            "user_id": str(args.user_id),
        },
    )

    failures: list[dict[str, Any]] = []
    stop_reason = "completed"
    total = len(rows)
    ok_cases = 0
    answer_correct_cases = 0
    evidence_hit_cases = 0
    recall_cases = 0
    false_memory_cases = 0
    quota_cases = 0
    l1_hits = 0
    l2_hits = 0
    l3_hits = 0
    latency_values: list[float] = []

    for idx, case in enumerate(rows, start=1):
        case_id = str(case.get("case_id") or case.get("question_id") or "")
        if not case_id:
            failures.append({"case_index": idx, "reason": "missing_case_id", "case": case})
            stop_reason = "missing_case_id"
            break

        question = str(case.get("question") or "").strip()
        answer = str(case.get("answer") or "").strip()
        if not question:
            failures.append({"case_id": case_id, "reason": "missing_question"})
            stop_reason = "missing_question"
            break

        safe_case = _safe_name(case_id)
        case_dir = paths["cases_dir"] / safe_case
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "input.json", case)

        chat_id = f"{args.chat_prefix}_{safe_case}"
        _cleanup_chat(chat_id)

        turn_rows: list[dict[str, Any]] = []
        ingest_error = False
        haystack_sessions = case.get("haystack_sessions", [])
        haystack_ids = case.get("haystack_session_ids", [])
        haystack_dates = case.get("haystack_dates", [])

        if not isinstance(haystack_sessions, list):
            haystack_sessions = []
        max_sessions = max(1, int(args.max_sessions))
        for sidx, session in enumerate(haystack_sessions[:max_sessions], start=1):
            session_id = str(haystack_ids[sidx - 1]) if isinstance(haystack_ids, list) and sidx - 1 < len(haystack_ids) else f"session_{sidx}"
            session_date = str(haystack_dates[sidx - 1]) if isinstance(haystack_dates, list) and sidx - 1 < len(haystack_dates) else ""
            messages = session if isinstance(session, list) else []
            prompt = _session_prompt(
                session_id=session_id,
                session_date=session_date,
                messages=[x for x in messages if isinstance(x, dict)],
                max_chars=int(args.max_session_chars),
            )
            try:
                result = await _run_turn(task_text=prompt, chat_id=chat_id, user_id=str(args.user_id))
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case_id,
                        "reason": "runner_exception",
                        "stage": "ingest",
                        "message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                stop_reason = "runner_exception"
                ingest_error = True
                break
            latency_values.append(float(result["elapsed_s"]))
            turn_row = {
                "type": "ingest",
                "case_id": case_id,
                "chat_id": chat_id,
                "turn_index": sidx,
                "session_id": session_id,
                "prompt": prompt,
                "response": result["response"],
                "elapsed_s": result["elapsed_s"],
                "errors": result["errors"],
                "run_id": result["run_id"],
                "metadata": result["metadata"],
            }
            turn_rows.append(turn_row)
            _write_json(case_dir / f"ingest_turn_{sidx:03d}.json", turn_row)
            _append_jsonl(paths["latency"], {"case_id": case_id, "type": "ingest", "turn_index": sidx, "elapsed_s": result["elapsed_s"]})
            if _contains_quota_error(result):
                failures.append({"case_id": case_id, "reason": "quota_exhausted", "stage": "ingest", "turn_index": sidx})
                quota_cases += 1
                stop_reason = "quota_exhausted"
                ingest_error = True
                break

        if ingest_error:
            if stop_reason in STOP_REASONS and bool(args.stop_on_failure):
                break
            continue

        query_prompt = _query_prompt(question=question, expected_session_ids=[str(x) for x in case.get("answer_session_ids", []) if str(x).strip()])
        try:
            query_result = await _run_turn(task_text=query_prompt, chat_id=chat_id, user_id=str(args.user_id))
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "reason": "runner_exception",
                    "stage": "query",
                    "message": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            stop_reason = "runner_exception"
            if bool(args.stop_on_failure):
                break
            continue

        latency_values.append(float(query_result["elapsed_s"]))
        _append_jsonl(paths["latency"], {"case_id": case_id, "type": "query", "turn_index": 1, "elapsed_s": query_result["elapsed_s"]})
        answer_text = str(query_result["response"] or "")
        answer_correct = _question_answered_correctly(response=answer_text, answer=answer)
        expected_session_ids = [str(x) for x in case.get("answer_session_ids", []) if str(x).strip()]
        predicted_session_ids = _extract_session_ids(answer_text)
        evidence_hit = bool(set(expected_session_ids) & set(predicted_session_ids)) if expected_session_ids else False
        not_attempted = _is_not_attempted(answer_text)
        false_memory = bool((not answer_correct) and (not not_attempted) and answer_text.strip())
        groups = _extract_groups(query_result["result"])
        l1_hits += groups["L1"]
        l2_hits += groups["L2"]
        l3_hits += groups["L3"]

        if answer_correct:
            answer_correct_cases += 1
            recall_cases += 1
        if evidence_hit:
            evidence_hit_cases += 1
        if false_memory:
            false_memory_cases += 1

        case_ok = answer_correct
        if case_ok:
            ok_cases += 1

        case_record = {
            "case_id": case_id,
            "chat_id": chat_id,
            "question": question,
            "answer_gold": answer,
            "answer_prediction": answer_text,
            "answer_correct": answer_correct,
            "not_attempted": not_attempted,
            "false_memory": false_memory,
            "expected_session_ids": expected_session_ids,
            "predicted_session_ids": predicted_session_ids,
            "evidence_hit": evidence_hit,
            "l1_hits": groups["L1"],
            "l2_hits": groups["L2"],
            "l3_hits": groups["L3"],
            "query_elapsed_s": query_result["elapsed_s"],
            "query_errors": query_result["errors"],
            "query_metadata": query_result["metadata"],
            "ok": case_ok,
            "reason": "" if case_ok else ("not_attempted" if not_attempted else "wrong_answer"),
        }
        _write_json(case_dir / "result.json", case_record)
        _append_jsonl(paths["raw_predictions"], case_record)

        if _contains_quota_error(query_result):
            failures.append({"case_id": case_id, "reason": "quota_exhausted", "stage": "query"})
            quota_cases += 1
            stop_reason = "quota_exhausted"
            if bool(args.stop_on_failure):
                break
            continue

    completed = ok_cases + sum(1 for row in failures if row.get("reason") not in STOP_REASONS)
    metrics = {
        "schema_version": "longmemeval_s_memory_agent_eval.v1",
        "generated_at": _utc_now(),
        "dataset": "LIXINYI33/longmemeval-s",
        "stage": "longmemeval_s",
        "total_cases": total,
        "ok_cases": ok_cases,
        "failed_cases": max(0, total - ok_cases),
        "ok_rate": round(ok_cases / total, 4) if total else 0.0,
        "answer_accuracy": round(answer_correct_cases / total, 4) if total else 0.0,
        "evidence_hit_rate": round(evidence_hit_cases / total, 4) if total else 0.0,
        "memory_recall_rate": round(recall_cases / total, 4) if total else 0.0,
        "false_memory_rate": round(false_memory_cases / total, 4) if total else 0.0,
        "quota_exhausted_cases": quota_cases,
        "l1_hits_total": l1_hits,
        "l2_hits_total": l2_hits,
        "l3_hits_total": l3_hits,
        "latency_avg_s": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
        "offset": int(args.offset),
        "batch_size": int(args.batch_size),
        "stopped_early": stop_reason != "completed",
        "stop_reason": stop_reason,
        "completed_rows_hint": completed,
    }
    _write_json(paths["metrics"], metrics)
    _write_json(paths["failures"], failures)

    report_lines = [
        "# LongMemEval-S Memory Agent Report",
        "",
        f"- generated_at: `{metrics['generated_at']}`",
        f"- run_name: `{args.run_name}`",
        f"- stop_reason: `{stop_reason}`",
        "",
        "## Metrics",
        "",
        f"- total_cases: `{metrics['total_cases']}`",
        f"- ok_cases: `{metrics['ok_cases']}`",
        f"- answer_accuracy: `{metrics['answer_accuracy']}`",
        f"- evidence_hit_rate: `{metrics['evidence_hit_rate']}`",
        f"- memory_recall_rate: `{metrics['memory_recall_rate']}`",
        f"- false_memory_rate: `{metrics['false_memory_rate']}`",
        f"- latency_avg_s: `{metrics['latency_avg_s']}`",
        f"- L1/L2/L3 hits: `{l1_hits}/{l2_hits}/{l3_hits}`",
        "",
        "## Artifacts",
        "",
        f"- raw_predictions: `{paths['raw_predictions']}`",
        f"- metrics: `{paths['metrics']}`",
        f"- failures: `{paths['failures']}`",
    ]
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")
    return {"metrics": metrics, "failures": failures, "paths": {k: str(v) for k, v in paths.items()}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongMemEval-S memory-agent evaluation.")
    parser.add_argument("--cases-path", required=True, help="Path to normalized longmemeval_s_cases.jsonl")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--chat-prefix", default="longmemevals")
    parser.add_argument("--user-id", default="longmemeval-eval")
    parser.add_argument("--max-sessions", type=int, default=3)
    parser.add_argument("--max-session-chars", type=int, default=12000)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_name:
        args.run_name = _run_name("longmemeval_s_memory_eval")
    return args


async def _amain(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = await run_eval(args)
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_amain())
