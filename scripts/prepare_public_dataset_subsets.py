# coding=utf-8
"""Fetch and normalize tiny public benchmark subsets for Self AI evals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from io import StringIO
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_OUT = ROOT_DIR / "benchmarks" / "public_subsets"
HUMANEVAL_PLUS_URL = "https://huggingface.co/datasets/evalplus/humanevalplus/resolve/main/test.jsonl"
BFCL_SIMPLE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_simple.json"
BFCL_SIMPLE_ANSWERS_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/BFCL_v3_simple.json"
SIMPLEQA_URL = "https://huggingface.co/datasets/basicv8vc/SimpleQA/resolve/main/simple_qa_test_set.csv"
SWEBENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
LONGMEMEVAL_S_DATASET = "LIXINYI33/longmemeval-s"


def _read_jsonl_url(url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with urllib.request.urlopen(url, timeout=120) as response:
        for raw in response:
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _read_text_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def _flatten_bfcl_question(value: Any) -> str:
    messages: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "").strip()
            if content:
                messages.append(f"{role}: {content}")
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return "\n".join(messages).strip()


def _bfcl_answer_options(answer_row: dict[str, Any]) -> list[list[dict[str, Any]]]:
    options: list[list[dict[str, Any]]] = []
    ground_truth = answer_row.get("ground_truth", [])
    if not isinstance(ground_truth, list):
        return options
    for option in ground_truth:
        if not isinstance(option, dict):
            continue
        calls: list[dict[str, Any]] = []
        for name, arguments in option.items():
            calls.append(
                {
                    "name": str(name),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )
        if calls:
            options.append(calls)
    return options


def prepare_humaneval_plus(*, out_dir: Path, limit: int) -> dict[str, Any]:
    raw_rows = _read_jsonl_url(HUMANEVAL_PLUS_URL)
    selected = raw_rows[: max(1, int(limit))]
    normalized: list[dict[str, Any]] = []
    raw_out: list[dict[str, Any]] = []
    for row in selected:
        task_id = str(row.get("task_id") or "")
        case = {
            "stage": "humaneval_plus",
            "case_id": task_id,
            "task_id": task_id,
            "prompt": str(row.get("prompt") or ""),
            "entry_point": str(row.get("entry_point") or ""),
            "test_code": str(row.get("test") or ""),
            "source_dataset": "EvalPlus HumanEval+",
            "source_url": HUMANEVAL_PLUS_URL,
        }
        normalized.append(case)
        raw_out.append(row)
    cases_path = out_dir / "humaneval_plus_cases.jsonl"
    raw_path = out_dir / "raw" / "humaneval_plus_raw.jsonl"
    manifest_path = out_dir / "humaneval_plus_manifest.json"
    _write_jsonl(cases_path, normalized)
    _write_jsonl(raw_path, raw_out)
    manifest = {
        "dataset": "EvalPlus HumanEval+",
        "source_url": HUMANEVAL_PLUS_URL,
        "case_count": len(normalized),
        "cases_path": str(cases_path),
        "raw_path": str(raw_path),
        "task_ids": [row["case_id"] for row in normalized],
    }
    _write_json(manifest_path, manifest)
    return manifest


def prepare_bfcl(*, out_dir: Path, limit: int) -> dict[str, Any]:
    raw_cases = _read_jsonl_url(BFCL_SIMPLE_URL)
    raw_answers = _read_jsonl_url(BFCL_SIMPLE_ANSWERS_URL)
    answers_by_id = {str(row.get("id") or ""): row for row in raw_answers}
    selected = raw_cases[: max(1, int(limit))]
    normalized: list[dict[str, Any]] = []
    raw_out: list[dict[str, Any]] = []
    for row in selected:
        case_id = str(row.get("id") or "")
        answer_row = answers_by_id.get(case_id, {})
        options = _bfcl_answer_options(answer_row)
        case = {
            "stage": "bfcl",
            "case_id": case_id,
            "prompt": _flatten_bfcl_question(row.get("question")),
            "messages": row.get("question", []),
            "functions": row.get("function", []),
            "expected_tool_call_options": options,
            "expected_tool_calls": options[0] if options else [],
            "source_dataset": "Berkeley Function Calling Leaderboard v3 simple",
            "source_url": BFCL_SIMPLE_URL,
            "answer_source_url": BFCL_SIMPLE_ANSWERS_URL,
        }
        normalized.append(case)
        raw_out.append({"case": row, "answer": answer_row})
    cases_path = out_dir / "bfcl_cases.jsonl"
    raw_path = out_dir / "raw" / "bfcl_raw.jsonl"
    manifest_path = out_dir / "bfcl_manifest.json"
    _write_jsonl(cases_path, normalized)
    _write_jsonl(raw_path, raw_out)
    manifest = {
        "dataset": "Berkeley Function Calling Leaderboard v3 simple",
        "source_url": BFCL_SIMPLE_URL,
        "answer_source_url": BFCL_SIMPLE_ANSWERS_URL,
        "case_count": len(normalized),
        "cases_path": str(cases_path),
        "raw_path": str(raw_path),
        "case_ids": [row["case_id"] for row in normalized],
    }
    _write_json(manifest_path, manifest)
    return manifest


def prepare_simpleqa(*, out_dir: Path, limit: int) -> dict[str, Any]:
    raw_text = _read_text_url(SIMPLEQA_URL)
    reader = csv.DictReader(StringIO(raw_text))
    raw_rows = [dict(row) for row in reader]
    selected = raw_rows[: max(1, int(limit))]
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        case_id = f"simpleqa_{idx}"
        normalized.append(
            {
                "stage": "simpleqa",
                "case_id": case_id,
                "question": str(row.get("problem") or "").strip(),
                "answer": str(row.get("answer") or "").strip(),
                "metadata": str(row.get("metadata") or "").strip(),
                "aliases": [],
                "source_dataset": "SimpleQA public test set",
                "source_url": SIMPLEQA_URL,
            }
        )
    cases_path = out_dir / "simpleqa_cases.jsonl"
    raw_path = out_dir / "raw" / "simpleqa_raw.jsonl"
    manifest_path = out_dir / "simpleqa_manifest.json"
    _write_jsonl(cases_path, normalized)
    _write_jsonl(raw_path, selected)
    manifest = {
        "dataset": "SimpleQA public test set",
        "source_url": SIMPLEQA_URL,
        "case_count": len(normalized),
        "cases_path": str(cases_path),
        "raw_path": str(raw_path),
        "case_ids": [row["case_id"] for row in normalized],
    }
    _write_json(manifest_path, manifest)
    return manifest


def prepare_swebench_lite(*, out_dir: Path, limit: int) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("datasets package is required to prepare SWE-bench Lite") from exc

    dataset = load_dataset(SWEBENCH_LITE_DATASET, split="test")
    selected = [dict(dataset[idx]) for idx in range(min(max(1, int(limit)), len(dataset)))]
    normalized: list[dict[str, Any]] = []
    for row in selected:
        instance_id = str(row.get("instance_id") or "")
        normalized.append(
            {
                "stage": "swebench_lite",
                "case_id": instance_id,
                "instance_id": instance_id,
                "repo": str(row.get("repo") or ""),
                "base_commit": str(row.get("base_commit") or ""),
                "problem_statement": str(row.get("problem_statement") or ""),
                "hints_text": str(row.get("hints_text") or ""),
                "version": str(row.get("version") or ""),
                "fail_to_pass": row.get("FAIL_TO_PASS", "[]"),
                "pass_to_pass": row.get("PASS_TO_PASS", "[]"),
                "test_patch": str(row.get("test_patch") or ""),
                "reference_patch": str(row.get("patch") or ""),
                "environment_setup_commit": str(row.get("environment_setup_commit") or ""),
                "source_dataset": "SWE-bench Lite",
                "source_url": f"https://huggingface.co/datasets/{SWEBENCH_LITE_DATASET}",
            }
        )
    cases_path = out_dir / "swebench_lite_cases.jsonl"
    raw_path = out_dir / "raw" / "swebench_lite_raw.jsonl"
    manifest_path = out_dir / "swebench_lite_manifest.json"
    _write_jsonl(cases_path, normalized)
    _write_jsonl(raw_path, selected)
    manifest = {
        "dataset": "SWE-bench Lite",
        "source": SWEBENCH_LITE_DATASET,
        "case_count": len(normalized),
        "cases_path": str(cases_path),
        "raw_path": str(raw_path),
        "case_ids": [row["case_id"] for row in normalized],
        "note": "reference_patch/test_patch are saved for validation and analysis; agent prompts must not include reference_patch.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def prepare_longmemeval_s(*, out_dir: Path, limit: int) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("datasets package is required to prepare LongMemEval-S") from exc

    dataset = load_dataset(LONGMEMEVAL_S_DATASET, split="train")
    selected = [dict(dataset[idx]) for idx in range(min(max(1, int(limit)), len(dataset)))]
    normalized: list[dict[str, Any]] = []

    for row in selected:
        case_id = str(row.get("question_id") or "")
        normalized.append(
            {
                "stage": "longmemeval_s",
                "case_id": case_id,
                "question_id": case_id,
                "question_type": str(row.get("question_type") or ""),
                "question": str(row.get("question") or ""),
                "answer": str(row.get("answer") or ""),
                "question_date": str(row.get("question_date") or ""),
                "haystack_dates": row.get("haystack_dates", []),
                "haystack_session_ids": row.get("haystack_session_ids", []),
                "haystack_sessions": row.get("haystack_sessions", []),
                "answer_session_ids": row.get("answer_session_ids", []),
                "source_dataset": "LongMemEval-S",
                "source": LONGMEMEVAL_S_DATASET,
                "source_url": f"https://huggingface.co/datasets/{LONGMEMEVAL_S_DATASET}",
            }
        )

    cases_path = out_dir / "longmemeval_s_cases.jsonl"
    raw_path = out_dir / "raw" / "longmemeval_s_raw.jsonl"
    manifest_path = out_dir / "longmemeval_s_manifest.json"
    _write_jsonl(cases_path, normalized)
    _write_jsonl(raw_path, selected)
    manifest = {
        "dataset": "LongMemEval-S",
        "source": LONGMEMEVAL_S_DATASET,
        "source_url": f"https://huggingface.co/datasets/{LONGMEMEVAL_S_DATASET}",
        "case_count": len(normalized),
        "cases_path": str(cases_path),
        "raw_path": str(raw_path),
        "case_ids": [row["case_id"] for row in normalized],
        "note": "Each case includes haystack sessions, dates, and answer_session_ids for long-memory evaluation.",
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized public benchmark subsets.")
    parser.add_argument(
        "--dataset",
        choices=["humaneval_plus", "bfcl", "simpleqa", "swebench_lite", "longmemeval_s"],
        default="humaneval_plus",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    if args.dataset == "humaneval_plus":
        manifest = prepare_humaneval_plus(out_dir=out_dir, limit=int(args.limit))
    elif args.dataset == "bfcl":
        manifest = prepare_bfcl(out_dir=out_dir, limit=int(args.limit))
    elif args.dataset == "simpleqa":
        manifest = prepare_simpleqa(out_dir=out_dir, limit=int(args.limit))
    elif args.dataset == "swebench_lite":
        manifest = prepare_swebench_lite(out_dir=out_dir, limit=int(args.limit))
    elif args.dataset == "longmemeval_s":
        manifest = prepare_longmemeval_s(out_dir=out_dir, limit=int(args.limit))
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
