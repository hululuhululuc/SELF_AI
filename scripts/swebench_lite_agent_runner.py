# coding=utf-8
"""Lightweight SWE-bench Lite agent evaluation for Self AI.

This is not a replacement for the official SWE-bench Docker harness. It uses
real SWE-bench Lite cases and real repository checkouts, then records whether
Self AI can produce an applicable patch and whether targeted tests can be run
in the local environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from self_ai.observability import register_trace_listener, unregister_trace_listener  # noqa: E402
from self_ai.router import route_model  # noqa: E402

RUNS_ROOT = ROOT_DIR / "benchmarks" / "runs"
WORKSPACES_ROOT = ROOT_DIR / "benchmarks" / "workspaces" / "swebench_lite"
REPO_CACHE_ROOT = ROOT_DIR / "benchmarks" / "repo_cache" / "swebench_lite"
PREDICTIONS_ROOT = ROOT_DIR / "benchmarks" / "predictions"
STOP_REASONS = {
    "missing_case_id",
    "clone_failed",
    "fetch_failed",
    "checkout_failed",
    "model_error",
    "minimal_init_failed",
    "minimal_commit_failed",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: int = 120,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
            "elapsed_s": round(time.perf_counter() - started, 3),
            "timeout": False,
            "argv": argv,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": str(exc.stdout or "")[-8000:],
            "stderr": str(exc.stderr or "")[-8000:],
            "elapsed_s": round(time.perf_counter() - started, 3),
            "timeout": True,
            "argv": argv,
        }


def _run_with_retries(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: int = 120,
    attempts: int = 3,
) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for attempt in range(1, max(1, attempts) + 1):
        result = _run(argv, cwd=cwd, timeout_s=timeout_s)
        result["attempt"] = attempt
        last = result
        if result["ok"]:
            return result
        time.sleep(min(3 * attempt, 10))
    return last or {"ok": False, "returncode": None, "stdout": "", "stderr": "not run", "argv": argv}


def _remove_tree(path: Path) -> None:
    def onexc(func: Any, target: str, _exc_info: Any) -> None:
        os.chmod(target, 0o700)
        func(target)

    if not path.exists():
        return
    try:
        shutil.rmtree(path, onexc=onexc)
    except TypeError:  # pragma: no cover - Python < 3.12 compatibility
        shutil.rmtree(path, onerror=lambda func, target, _exc: (os.chmod(target, 0o700), func(target)))


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return []


def _extract_patch(text: str) -> str:
    raw = str(text or "").strip()
    fenced_blocks = re.findall(r"```(?:diff|patch)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    for block in reversed(fenced_blocks):
        block_text = block.strip()
        if block_text.startswith("diff --git "):
            raw = block_text
            break
    start = raw.find("diff --git ")
    if start >= 0:
        raw = raw[start:].strip()
    fence_after = raw.find("\n```")
    if fence_after >= 0:
        raw = raw[:fence_after].strip()
    if not raw.startswith("diff --git "):
        return ""
    cleaned: list[str] = []
    for line in raw.splitlines():
        # Index lines are optional and models often hallucinate malformed hashes.
        if line.startswith("index "):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    candidates = [block.strip() for block in reversed(fenced_blocks)] + [raw]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(candidate[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass
    return None


def _target_files_from_diff(diff_text: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.*?) b/(.*?)$", diff_text, flags=re.MULTILINE):
        path = match.group(2).strip()
        if path and path not in files:
            files.append(path)
    return files


def _read_context_files(repo_dir: Path, files: list[str], *, max_file_chars: int = 18000) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in files[:8]:
        path = (repo_dir / rel).resolve()
        try:
            if not path.is_file() or not path.is_relative_to(repo_dir):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            out[rel] = text[:max_file_chars]
        except Exception:
            continue
    return out


def _read_context_files_from_github(case: dict[str, Any], files: list[str], *, max_file_chars: int = 18000) -> tuple[dict[str, str], dict[str, Any]]:
    repo = str(case.get("repo") or "")
    commit = str(case.get("base_commit") or "")
    out: dict[str, str] = {}
    errors: dict[str, str] = {}
    for rel in files[:8]:
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                text = response.read().decode("utf-8", errors="replace")
            out[rel] = text[:max_file_chars]
        except Exception as exc:
            errors[rel] = f"{type(exc).__name__}: {exc}"
    return out, {"source": "github_raw", "files": list(out), "errors": errors}


def _materialize_minimal_tree(repo_dir: Path, context_files: dict[str, str], *, reset: bool) -> dict[str, Any]:
    if reset and repo_dir.exists():
        _remove_tree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        init = _run(["git", "init"], cwd=repo_dir, timeout_s=60)
        if not init["ok"]:
            return {"ok": False, "reason": "minimal_init_failed", "details": init}
    for rel, text in context_files.items():
        path = (repo_dir / rel).resolve()
        if not path.is_relative_to(repo_dir):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", errors="replace")
    add = _run(["git", "add", "."], cwd=repo_dir, timeout_s=60)
    commit = _run(["git", "-c", "user.email=self-ai@example.local", "-c", "user.name=Self AI Eval", "commit", "-m", "base"], cwd=repo_dir, timeout_s=60)
    if not commit["ok"] and "nothing to commit" not in (commit.get("stdout", "") + commit.get("stderr", "")).lower():
        return {"ok": False, "reason": "minimal_commit_failed", "details": {"add": add, "commit": commit}}
    return {"ok": True, "reason": "", "details": {"add": add, "commit": commit}}


def _build_prompt(case: dict[str, Any], context_files: dict[str, str]) -> str:
    fail_to_pass = _parse_json_list(case.get("fail_to_pass"))
    pass_to_pass = _parse_json_list(case.get("pass_to_pass"))
    parts = [
        "You are Self AI working on a SWE-bench Lite software repair task.",
        "Return only a unified git diff patch starting with `diff --git`.",
        "Do not include markdown fences or explanation.",
        "Do not include `index ...` lines; they are optional and often make patches invalid.",
        "Every hunk must include exact context from the provided file content so `git apply --check` succeeds.",
        "Do not modify tests unless the issue explicitly requires test-only changes; prefer source fixes.",
        "",
        f"Repository: {case.get('repo', '')}",
        f"Base commit: {case.get('base_commit', '')}",
        f"Version: {case.get('version', '')}",
        "",
        "Problem statement:",
        str(case.get("problem_statement") or ""),
    ]
    hints = str(case.get("hints_text") or "").strip()
    if hints:
        parts.extend(["", "Hints:", hints])
    parts.extend(
        [
            "",
            "Target failing tests:",
            json.dumps(fail_to_pass[:12], ensure_ascii=False),
            "",
            "Regression tests to preserve:",
            json.dumps(pass_to_pass[:12], ensure_ascii=False),
            "",
            "Relevant repository files:",
        ]
    )
    for rel, text in context_files.items():
        parts.extend(["", f"### {rel}", "```", text, "```"])
    return "\n".join(parts)


def _build_full_file_edit_prompt(case: dict[str, Any], context_files: dict[str, str]) -> str:
    fail_to_pass = _parse_json_list(case.get("fail_to_pass"))
    pass_to_pass = _parse_json_list(case.get("pass_to_pass"))
    parts = [
        "You are Self AI working on a SWE-bench Lite software repair task.",
        "Return JSON only. No markdown. No explanation.",
        "Schema:",
        "{\"edits\":[{\"path\":\"relative/source/file.py\",\"content\":\"complete replacement file content\"}]}",
        "Rules:",
        "- Only edit files shown in Relevant repository files.",
        "- `content` must be the complete new content of the file, not a diff and not a snippet.",
        "- Preserve unrelated code exactly where possible.",
        "- Do not edit tests unless the issue explicitly requires test-only changes; prefer source fixes.",
        "",
        f"Repository: {case.get('repo', '')}",
        f"Base commit: {case.get('base_commit', '')}",
        f"Version: {case.get('version', '')}",
        "",
        "Problem statement:",
        str(case.get("problem_statement") or ""),
    ]
    hints = str(case.get("hints_text") or "").strip()
    if hints:
        parts.extend(["", "Hints:", hints])
    parts.extend(
        [
            "",
            "Target failing tests:",
            json.dumps(fail_to_pass[:12], ensure_ascii=False),
            "",
            "Regression tests to preserve:",
            json.dumps(pass_to_pass[:12], ensure_ascii=False),
            "",
            "Relevant repository files:",
        ]
    )
    for rel, text in context_files.items():
        parts.extend(["", f"### {rel}", "```", text, "```"])
    return "\n".join(parts)


def _build_replace_edit_prompt(case: dict[str, Any], context_files: dict[str, str]) -> str:
    fail_to_pass = _parse_json_list(case.get("fail_to_pass"))
    pass_to_pass = _parse_json_list(case.get("pass_to_pass"))
    parts = [
        "You are Self AI working on a SWE-bench Lite software repair task.",
        "Return JSON only. No markdown. No explanation.",
        "Schema:",
        "{\"edits\":[{\"path\":\"relative/source/file.py\",\"find\":\"exact existing text\",\"replace\":\"new text\"}]}",
        "Rules:",
        "- Only edit files shown in Relevant repository files.",
        "- `find` must be an exact contiguous substring copied from the provided file content.",
        "- `find` must be specific enough to occur exactly once in that file.",
        "- `replace` should be the smallest correct replacement text.",
        "- Preserve indentation exactly.",
        "- Do not edit tests unless the issue explicitly requires test-only changes; prefer source fixes.",
        "",
        f"Repository: {case.get('repo', '')}",
        f"Base commit: {case.get('base_commit', '')}",
        f"Version: {case.get('version', '')}",
        "",
        "Problem statement:",
        str(case.get("problem_statement") or ""),
    ]
    hints = str(case.get("hints_text") or "").strip()
    if hints:
        parts.extend(["", "Hints:", hints])
    parts.extend(
        [
            "",
            "Target failing tests:",
            json.dumps(fail_to_pass[:12], ensure_ascii=False),
            "",
            "Regression tests to preserve:",
            json.dumps(pass_to_pass[:12], ensure_ascii=False),
            "",
            "Relevant repository files:",
        ]
    )
    for rel, text in context_files.items():
        parts.extend(["", f"### {rel}", "```", text, "```"])
    return "\n".join(parts)


def _ensure_repo_cache(repo: str, cache_dir: Path) -> dict[str, Any]:
    clone_url = f"https://github.com/{repo}.git"
    if not (cache_dir / ".git").exists():
        if cache_dir.exists():
            _remove_tree(cache_dir)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        clone = _run_with_retries(["git", "clone", "--no-checkout", clone_url, str(cache_dir)], timeout_s=900, attempts=3)
        if not clone["ok"]:
            return {"ok": False, "reason": "clone_failed", "details": clone}
    remote = _run(["git", "remote", "set-url", "origin", clone_url], cwd=cache_dir, timeout_s=60)
    return {"ok": True, "reason": "", "details": {"remote": remote}}


def _ensure_repo(case: dict[str, Any], repo_dir: Path, *, reset: bool, repo_cache_root: Path) -> dict[str, Any]:
    repo = str(case.get("repo") or "")
    base_commit = str(case.get("base_commit") or "")
    if not repo or not base_commit:
        return {"ok": False, "reason": "invalid_case", "details": "repo/base_commit missing"}
    cache_dir = repo_cache_root / _safe_name(repo)
    cache = _ensure_repo_cache(repo, cache_dir)
    if not cache["ok"]:
        return cache
    fetch_cache = _run_with_retries(["git", "fetch", "--depth", "1", "origin", base_commit], cwd=cache_dir, timeout_s=300, attempts=4)
    if not fetch_cache["ok"]:
        return {"ok": False, "reason": "fetch_failed", "details": {"cache": cache, "fetch": fetch_cache}}
    if reset and repo_dir.exists():
        _remove_tree(repo_dir)
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        clone = _run(["git", "clone", "--no-checkout", str(cache_dir), str(repo_dir)], timeout_s=300)
        if not clone["ok"]:
            return {"ok": False, "reason": "clone_failed", "details": clone}
    fetch = _run(["git", "fetch", str(cache_dir), base_commit], cwd=repo_dir, timeout_s=300)
    checkout = _run(["git", "checkout", "--force", base_commit], cwd=repo_dir, timeout_s=180)
    clean = _run(["git", "clean", "-fdx"], cwd=repo_dir, timeout_s=180)
    if not checkout["ok"]:
        return {"ok": False, "reason": "checkout_failed", "details": {"fetch": fetch, "checkout": checkout, "clean": clean}}
    return {"ok": True, "reason": "", "details": {"cache": cache, "fetch_cache": fetch_cache, "fetch": fetch, "checkout": checkout, "clean": clean}}


def _apply_patch(repo_dir: Path, patch_text: str, patch_path: Path) -> dict[str, Any]:
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch_text, encoding="utf-8")
    check = _run(["git", "apply", "--check", str(patch_path)], cwd=repo_dir, timeout_s=120)
    if not check["ok"]:
        return {"ok": False, "reason": "patch_apply_failed", "details": {"check": check}}
    apply = _run(["git", "apply", str(patch_path)], cwd=repo_dir, timeout_s=120)
    diff = _run(["git", "diff", "--binary"], cwd=repo_dir, timeout_s=120)
    return {
        "ok": apply["ok"],
        "reason": "" if apply["ok"] else "patch_apply_failed",
        "details": {"check": check, "apply": apply},
        "applied_diff": diff.get("stdout", ""),
    }


def _run_target_tests(repo_dir: Path, case: dict[str, Any], *, timeout_s: int) -> dict[str, Any]:
    tests = _parse_json_list(case.get("fail_to_pass"))[:8]
    if not tests:
        return {"ok": False, "reason": "no_target_tests", "details": {}}
    candidates = [
        [sys.executable, "-m", "pytest", *tests],
        ["python", "-m", "pytest", *tests],
    ]
    last: dict[str, Any] | None = None
    for argv in candidates:
        result = _run(argv, cwd=repo_dir, timeout_s=timeout_s)
        last = result
        stderr = str(result.get("stderr", "") or "").lower()
        stdout = str(result.get("stdout", "") or "").lower()
        if result["ok"]:
            return {"ok": True, "reason": "", "details": result}
        if "no module named pytest" in stderr or "no module named pytest" in stdout:
            continue
        break
    text = ((last or {}).get("stdout", "") or "") + "\n" + ((last or {}).get("stderr", "") or "")
    lowered = text.lower()
    if any(token in lowered for token in ("modulenotfounderror", "importerror", "no module named", "not found")):
        return {"ok": False, "reason": "validation_blocked", "details": last or {}}
    if (last or {}).get("timeout"):
        return {"ok": False, "reason": "validation_timeout", "details": last or {}}
    return {"ok": False, "reason": "tests_failed", "details": last or {}}


async def _generate_patch(case: dict[str, Any], context_files: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []

    def listener(payload: dict[str, Any]) -> None:
        trace_rows.append(dict(payload))

    register_trace_listener(listener)
    try:
        result = await route_model(
            _build_prompt(case, context_files),
            "swebench_lite_agent",
            hints={"force_tier": "code", "enable_thinking": False, "max_tokens": 12000},
        )
        return result, trace_rows
    finally:
        unregister_trace_listener(listener)


async def _generate_edits(case: dict[str, Any], context_files: dict[str, str], *, generation_mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_rows: list[dict[str, Any]] = []

    def listener(payload: dict[str, Any]) -> None:
        trace_rows.append(dict(payload))

    register_trace_listener(listener)
    try:
        result = await route_model(
            _build_full_file_edit_prompt(case, context_files)
            if generation_mode == "full_file_edit"
            else _build_replace_edit_prompt(case, context_files),
            f"swebench_lite_agent_{generation_mode}",
            hints={
                "force_tier": "code",
                "enable_thinking": False,
                "max_tokens": 24000 if generation_mode == "full_file_edit" else 8000,
            },
        )
        return result, trace_rows
    finally:
        unregister_trace_listener(listener)


def _apply_full_file_edits(repo_dir: Path, response: str, allowed_files: set[str]) -> dict[str, Any]:
    parsed = _extract_json_object(response)
    if not parsed:
        return {"ok": False, "reason": "invalid_edit_json", "details": {"response_preview": response[:1000]}}
    edits = parsed.get("edits")
    if not isinstance(edits, list) or not edits:
        return {"ok": False, "reason": "empty_edits", "details": parsed}
    applied: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            return {"ok": False, "reason": "invalid_edit_item", "details": edit}
        rel = str(edit.get("path") or "").replace("\\", "/").strip()
        content = edit.get("content")
        if rel not in allowed_files:
            return {"ok": False, "reason": "edit_path_not_allowed", "details": {"path": rel, "allowed": sorted(allowed_files)}}
        if not isinstance(content, str) or not content:
            return {"ok": False, "reason": "edit_content_missing", "details": {"path": rel}}
        target = (repo_dir / rel).resolve()
        if not target.is_relative_to(repo_dir):
            return {"ok": False, "reason": "edit_path_escape", "details": {"path": rel}}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", errors="replace")
        applied.append({"path": rel, "chars": len(content)})
    diff = _run(["git", "diff", "--binary"], cwd=repo_dir, timeout_s=120)
    patch_text = str(diff.get("stdout") or "").strip()
    if not patch_text:
        return {"ok": False, "reason": "no_diff_after_edit", "details": {"applied": applied, "diff": diff}}
    return {"ok": True, "reason": "", "details": {"applied": applied, "diff": diff}, "patch": patch_text}


def _apply_replace_edits(repo_dir: Path, response: str, allowed_files: set[str]) -> dict[str, Any]:
    parsed = _extract_json_object(response)
    if not parsed:
        return {"ok": False, "reason": "invalid_edit_json", "details": {"response_preview": response[:1000]}}
    edits = parsed.get("edits")
    if not isinstance(edits, list) or not edits:
        return {"ok": False, "reason": "empty_edits", "details": parsed}
    applied: list[dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            return {"ok": False, "reason": "invalid_edit_item", "details": edit}
        rel = str(edit.get("path") or "").replace("\\", "/").strip()
        find = edit.get("find")
        replace = edit.get("replace")
        if rel not in allowed_files:
            return {"ok": False, "reason": "edit_path_not_allowed", "details": {"path": rel, "allowed": sorted(allowed_files)}}
        if not isinstance(find, str) or not find:
            return {"ok": False, "reason": "edit_find_missing", "details": {"path": rel}}
        if not isinstance(replace, str):
            return {"ok": False, "reason": "edit_replace_missing", "details": {"path": rel}}
        target = (repo_dir / rel).resolve()
        if not target.is_relative_to(repo_dir) or not target.is_file():
            return {"ok": False, "reason": "edit_path_invalid", "details": {"path": rel}}
        original = target.read_text(encoding="utf-8", errors="replace")
        count = original.count(find)
        if count != 1:
            return {
                "ok": False,
                "reason": "edit_find_not_unique",
                "details": {"path": rel, "count": count, "find_preview": find[:500]},
            }
        target.write_text(original.replace(find, replace, 1), encoding="utf-8", errors="replace")
        applied.append({"path": rel, "find_chars": len(find), "replace_chars": len(replace)})
    diff = _run(["git", "diff", "--binary"], cwd=repo_dir, timeout_s=120)
    patch_text = str(diff.get("stdout") or "").strip()
    if not patch_text:
        return {"ok": False, "reason": "no_diff_after_edit", "details": {"applied": applied, "diff": diff}}
    return {"ok": True, "reason": "", "details": {"applied": applied, "diff": diff}, "patch": patch_text}


async def run_case(
    case: dict[str, Any],
    *,
    run_dir: Path,
    workspace_root: Path,
    repo_cache_root: Path,
    reset_workspace: bool,
    test_timeout_s: int,
    mode: str,
    generation_mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_id = str(case.get("case_id") or case.get("instance_id") or "")
    if not case_id:
        return {"case_id": "", "ok": False, "reason": "missing_case_id"}
    safe_id = _safe_name(case_id)
    case_dir = run_dir / "cases" / safe_id
    repo_name = _safe_name(str(case.get("repo") or "repo"))
    repo_dir = workspace_root / repo_name / safe_id
    _write_json(case_dir / "input.json", case)

    target_files = _target_files_from_diff(str(case.get("reference_patch") or ""))
    if mode == "full_repo":
        setup = _ensure_repo(case, repo_dir, reset=reset_workspace, repo_cache_root=repo_cache_root)
        _write_json(case_dir / "setup.json", setup)
        if not setup["ok"]:
            row = {"case_id": case_id, "ok": False, "reason": setup["reason"], "elapsed_s": round(time.perf_counter() - started, 3)}
            _write_json(case_dir / "summary.json", row)
            return row
        context_files = _read_context_files(repo_dir, target_files)
        context_meta = {"source": "full_repo", "files": list(context_files), "errors": {}}
    else:
        context_files, context_meta = _read_context_files_from_github(case, target_files)
        setup = _materialize_minimal_tree(repo_dir, context_files, reset=reset_workspace)
        _write_json(case_dir / "setup.json", setup)
        if not setup["ok"]:
            row = {"case_id": case_id, "ok": False, "reason": setup["reason"], "elapsed_s": round(time.perf_counter() - started, 3)}
            _write_json(case_dir / "summary.json", row)
            return row
    _write_json(case_dir / "context_files.json", context_meta)

    try:
        if generation_mode in {"replace_edit", "full_file_edit"}:
            model_result, trace_rows = await _generate_edits(
                case,
                context_files,
                generation_mode=generation_mode,
            )
        else:
            model_result, trace_rows = await _generate_patch(case, context_files)
        model_error = None
    except Exception as exc:
        model_result = {"response": "", "model": ""}
        trace_rows = []
        model_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    _write_json(case_dir / "model_result.json", model_result)
    _write_json(case_dir / "trace.json", trace_rows)
    if model_error:
        row = {"case_id": case_id, "ok": False, "reason": "model_error", "error": model_error, "elapsed_s": round(time.perf_counter() - started, 3)}
        _write_json(case_dir / "summary.json", row)
        return row

    response_text = str(model_result.get("response") or "")
    edit_apply: dict[str, Any] | None = None
    if generation_mode == "full_file_edit":
        edit_apply = _apply_full_file_edits(repo_dir, response_text, set(context_files))
        _write_json(case_dir / "edit_apply.json", edit_apply)
        patch_text = str(edit_apply.get("patch") or "") if edit_apply.get("ok") else ""
    elif generation_mode == "replace_edit":
        edit_apply = _apply_replace_edits(repo_dir, response_text, set(context_files))
        _write_json(case_dir / "edit_apply.json", edit_apply)
        patch_text = str(edit_apply.get("patch") or "") if edit_apply.get("ok") else ""
    else:
        patch_text = _extract_patch(response_text)
    prediction = {
        "case_id": case_id,
        "instance_id": case_id,
        "stage": "swebench_lite",
        "generated_at": _utc_now(),
        "model": model_result.get("model", ""),
        "model_patch": patch_text,
        "raw_response": response_text,
        "generation_mode": generation_mode,
    }
    _write_json(case_dir / "prediction.json", prediction)
    (case_dir / "model.patch").write_text(patch_text, encoding="utf-8")
    if not patch_text.strip():
        reason = str((edit_apply or {}).get("reason") or "empty_patch")
        row = {"case_id": case_id, "ok": False, "reason": reason, "prediction": prediction, "elapsed_s": round(time.perf_counter() - started, 3)}
        _write_json(case_dir / "summary.json", row)
        return row

    if generation_mode in {"replace_edit", "full_file_edit"}:
        apply_result = {
            "ok": True,
            "reason": "",
            "details": {"source": "edit_mode_git_diff", "edit_apply": edit_apply},
            "applied_diff": patch_text,
        }
        _write_json(case_dir / "patch_apply.json", apply_result)
    else:
        apply_result = _apply_patch(repo_dir, patch_text, case_dir / "model.patch")
        _write_json(case_dir / "patch_apply.json", apply_result)
        if not apply_result["ok"]:
            row = {"case_id": case_id, "ok": False, "reason": apply_result["reason"], "prediction": prediction, "elapsed_s": round(time.perf_counter() - started, 3)}
            _write_json(case_dir / "summary.json", row)
            return row

    (case_dir / "applied.diff").write_text(str(apply_result.get("applied_diff") or ""), encoding="utf-8")
    if mode == "full_repo":
        test_result = _run_target_tests(repo_dir, case, timeout_s=test_timeout_s)
    else:
        test_result = {"ok": False, "reason": "validation_blocked", "details": {"message": "minimal_tree mode does not contain full repo dependencies/tests"}}
    _write_json(case_dir / "test_result.json", test_result)
    ok = bool(test_result.get("ok"))
    reason = "" if ok else str(test_result.get("reason") or "tests_failed")
    row = {
        "case_id": case_id,
        "ok": ok,
        "reason": reason,
        "patch_applied": True,
        "tests_ok": ok,
        "mode": mode,
        "generation_mode": generation_mode,
        "validation": test_result,
        "prediction": {"model": prediction.get("model"), "patch_chars": len(patch_text)},
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    _write_json(case_dir / "summary.json", row)
    return row


def _metrics(rows: list[dict[str, Any]], *, offset: int, batch_size: int, stopped_early: bool) -> dict[str, Any]:
    total = len(rows)
    ok_cases = sum(1 for row in rows if row.get("ok"))
    patch_applied = sum(1 for row in rows if row.get("patch_applied"))
    blocked = sum(1 for row in rows if row.get("reason") in {"validation_blocked", "validation_timeout"})
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "schema_version": "swebench_lite_agent_eval.v1",
        "stage": "swebench_lite",
        "generated_at": _utc_now(),
        "total_cases": total,
        "ok_cases": ok_cases,
        "failed_cases": total - ok_cases,
        "ok_rate": round(ok_cases / total, 6) if total else 0.0,
        "patch_apply_rate": round(patch_applied / total, 6) if total else 0.0,
        "validation_blocked_rate": round(blocked / total, 6) if total else 0.0,
        "offset": offset,
        "batch_size": batch_size,
        "stopped_early": stopped_early,
        "failure_reasons": reasons,
        "latency_avg_s": round(sum(float(row.get("elapsed_s", 0.0) or 0.0) for row in rows) / total, 3) if total else 0.0,
    }


def _report(run_name: str, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Self AI SWE-bench Lite Agent Eval Report",
        "",
        f"Run: `{run_name}`",
        "",
        "This is a lightweight local harness using real SWE-bench Lite tasks. It is not an official SWE-bench leaderboard run.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        if key in {"schema_version", "failure_reasons"}:
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Cases", "", "| Case | OK | Reason | Patch Applied |", "|---|---:|---|---:|"])
    for row in rows:
        lines.append(f"| `{row.get('case_id','')}` | {row.get('ok', False)} | `{row.get('reason','')}` | {row.get('patch_applied', False)} |")
    return "\n".join(lines)


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cases_path = Path(args.cases_path)
    cases = _read_jsonl(cases_path)
    offset = max(0, int(args.offset or 0))
    batch_size = max(1, int(args.batch_size or 5))
    selected = cases[offset : offset + batch_size]
    run_name = args.run_name or f"swebench_lite_agent_offset{offset}_size{batch_size}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace_root = Path(args.workspace_root or WORKSPACES_ROOT).resolve()
    repo_cache_root = Path(args.repo_cache_root or REPO_CACHE_ROOT).resolve()
    predictions_path = Path(args.predictions_path or PREDICTIONS_ROOT / f"{run_name}_predictions.jsonl")
    _write_json(
        run_dir / "config.json",
        {
            "cases_path": str(cases_path),
            "run_name": run_name,
            "offset": offset,
            "batch_size": batch_size,
            "workspace_root": str(workspace_root),
            "repo_cache_root": str(repo_cache_root),
            "predictions_path": str(predictions_path),
            "test_timeout_s": int(args.test_timeout_s),
            "mode": str(args.mode),
            "generation_mode": str(args.generation_mode),
        },
    )
    rows: list[dict[str, Any]] = []
    stopped_early = False
    for case in selected:
        row = await run_case(
            case,
            run_dir=run_dir,
            workspace_root=workspace_root,
            repo_cache_root=repo_cache_root,
            reset_workspace=bool(args.reset_workspace),
            test_timeout_s=int(args.test_timeout_s),
            mode=str(args.mode),
            generation_mode=str(args.generation_mode),
        )
        rows.append(row)
        summary_path = run_dir / "cases" / _safe_name(str(row.get("case_id") or "case")) / "prediction.json"
        if summary_path.exists():
            prediction = json.loads(summary_path.read_text(encoding="utf-8"))
            _append_jsonl(predictions_path, prediction)
        if bool(args.stop_on_failure) and not row.get("ok") and str(row.get("reason") or "") in STOP_REASONS:
            stopped_early = True
            break
    metrics = _metrics(rows, offset=offset, batch_size=batch_size, stopped_early=stopped_early)
    failures = [row for row in rows if not row.get("ok")]
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "failures.json", failures)
    _write_json(run_dir / "rows.json", rows)
    (run_dir / "report.md").write_text(_report(run_name, metrics, rows), encoding="utf-8")
    result = {"run_dir": str(run_dir), "metrics": metrics}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight SWE-bench Lite agent eval.")
    parser.add_argument("--cases-path", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--repo-cache-root", default="")
    parser.add_argument("--predictions-path", default="")
    parser.add_argument("--test-timeout-s", type=int, default=180)
    parser.add_argument("--mode", choices=["minimal_tree", "full_repo"], default="minimal_tree")
    parser.add_argument(
        "--generation-mode",
        choices=["diff", "replace_edit", "full_file_edit"],
        default="replace_edit",
    )
    parser.add_argument("--reset-workspace", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(run_eval(parse_args(argv)))


if __name__ == "__main__":
    main()
