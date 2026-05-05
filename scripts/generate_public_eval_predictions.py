# coding=utf-8
"""Generate prediction JSONL for normalized public eval cases using Self AI."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from self_ai.main import run_autonomy_workflow  # noqa: E402
from self_ai.observability import register_trace_listener, unregister_trace_listener  # noqa: E402
from self_ai.router import route_model  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"JSONL row must be object at {path}:{line_no}")
        rows.append(parsed)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _safe_case_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _extract_code(text: str) -> str:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return raw


def _extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None
    return None


def _coerce_tool_calls(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        parsed = _extract_json(value)
        return _coerce_tool_calls(parsed)
    if isinstance(value, dict):
        if "tool_calls" in value:
            return _coerce_tool_calls(value.get("tool_calls"))
        if "calls" in value:
            return _coerce_tool_calls(value.get("calls"))
        name = value.get("name") or value.get("function") or value.get("tool_name")
        args = value.get("arguments", value.get("args", {}))
        if isinstance(name, str) and name.strip():
            return [{"name": name.strip(), "arguments": args if isinstance(args, dict) else {}}]
        return []
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for item in value:
            out.extend(_coerce_tool_calls(item))
        return out
    return []


def _build_prompt(case: dict[str, Any]) -> str:
    stage = str(case.get("stage") or "")
    if stage == "humaneval_plus":
        return (
            "Complete the following Python function for a HumanEval-style benchmark.\n"
            "Return only valid Python code. Do not include markdown fences or explanation.\n\n"
            f"{case.get('prompt', '')}"
        )
    if stage == "bfcl":
        return (
            "You are solving a Berkeley Function Calling Leaderboard item.\n"
            "Select the correct function call(s) for the user request.\n"
            "Return JSON only in this exact shape:\n"
            "{\"tool_calls\":[{\"name\":\"function_name\",\"arguments\":{}}]}\n"
            "Do not execute the function. Do not include explanation.\n\n"
            f"User request:\n{case.get('prompt', '')}\n\n"
            "Available functions JSON:\n"
            f"{json.dumps(case.get('functions', []), ensure_ascii=False)}"
        )
    if stage == "simpleqa":
        return (
            "Answer this SimpleQA factual question.\n"
            "Return only the shortest exact answer string. Do not include explanation.\n\n"
            f"Question: {case.get('question', '')}"
        )
    raise ValueError(f"prediction generation for stage is not implemented: {stage}")


async def generate(args: argparse.Namespace) -> dict[str, Any]:
    cases_path = Path(args.cases_path)
    out_path = Path(args.out_path)
    artifacts_dir = Path(args.artifacts_dir or out_path.parent / (out_path.stem + "_cases"))
    cases = _read_jsonl(cases_path)
    start = max(0, int(args.offset or 0))
    selected = cases[start:]
    if args.batch_size:
        selected = selected[: max(1, int(args.batch_size))]
    elif args.limit:
        selected = selected[: max(1, int(args.limit))]
    existing: set[str] = set()
    if out_path.exists() and bool(args.resume):
        for row in _read_jsonl(out_path):
            case_id = str(row.get("case_id") or "")
            if case_id:
                existing.add(case_id)

    summaries: list[dict[str, Any]] = []
    stopped_early = False
    for case in selected:
        case_id = str(case.get("case_id") or case.get("task_id") or "")
        if not case_id:
            summaries.append({"case_id": "", "ok": False, "reason": "case_id_missing"})
            stopped_early = True
            break
        if case_id in existing:
            continue
        safe_id = _safe_case_id(case_id)
        trace_rows: list[dict[str, Any]] = []

        def listener(payload: dict[str, Any]) -> None:
            trace_rows.append(dict(payload))

        started = time.perf_counter()
        result: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        register_trace_listener(listener)
        try:
            stage = str(case.get("stage", "") or "")
            if stage in {"bfcl", "simpleqa"}:
                result = await route_model(
                    _build_prompt(case),
                    "public_eval",
                    hints={
                        "force_tier": "balanced",
                        "enable_thinking": False,
                        "max_tokens": 2048,
                    },
                )
            else:
                result = await run_autonomy_workflow(
                    _build_prompt(case),
                    session_id=f"public-eval-{safe_id}",
                    chat_id=f"public-eval-{safe_id}",
                    user_id="public-eval",
                )
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            unregister_trace_listener(listener)
        elapsed_s = time.perf_counter() - started
        response = str((result or {}).get("response", "") or "")
        stage = str(case.get("stage", "") or "")
        row = {
            "case_id": case_id,
            "stage": stage,
            "generated_at": _utc_now(),
            "elapsed_s": round(elapsed_s, 3),
            "response": response,
            "model": (result or {}).get("model", ""),
            "run_id": (result or {}).get("run_id", ""),
            "errors": (result or {}).get("errors", []),
            "error": error,
        }
        if stage == "humaneval_plus":
            row["completion"] = _extract_code(response)
        elif stage == "bfcl":
            row["tool_calls"] = _coerce_tool_calls(response)
        elif stage == "simpleqa":
            row["answer"] = response.strip().strip("\"'`")
        _append_jsonl(out_path, row)
        case_dir = artifacts_dir / safe_id
        _write_json(case_dir / "input.json", case)
        _write_json(case_dir / "prediction.json", row)
        _write_json(case_dir / "result.json", result or {})
        _write_json(case_dir / "trace.json", trace_rows)
        summary = {
            "case_id": case_id,
            "ok": error is None and not bool((result or {}).get("errors", [])),
            "elapsed_s": round(elapsed_s, 3),
            "model": row["model"],
            "error": error,
            "structured_errors": (result or {}).get("errors", []),
            "trace_events": len(trace_rows),
        }
        _write_json(case_dir / "summary.json", summary)
        summaries.append(summary)
        if bool(args.stop_on_failure) and not summary["ok"]:
            stopped_early = True
            break
    manifest = {
        "schema_version": "public_eval_predictions.v1",
        "generated_at": _utc_now(),
        "cases_path": str(cases_path),
        "out_path": str(out_path),
        "artifacts_dir": str(artifacts_dir),
        "offset": start,
        "batch_size": int(args.batch_size or 0),
        "selected_case_count": len(selected),
        "generated_case_count": len(summaries),
        "stopped_early": stopped_early,
        "summaries": summaries,
    }
    _write_json(out_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prediction JSONL for public eval cases.")
    parser.add_argument("--cases-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--artifacts-dir", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
