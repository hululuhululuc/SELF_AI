# coding=utf-8
"""Self AI Agent Cockpit frontend."""

from __future__ import annotations

import asyncio
import html
import json
import queue
import threading
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

from self_ai import run_autonomy_workflow, run_tool_once
from self_ai.frontend_formatting import (
    format_agent_turns,
    format_engine_timeline,
    format_errors,
    format_final_response,
    format_memory_context,
    format_quality_gate,
    format_review_reports,
    format_revision_summary,
    format_runtime_signals,
    format_run_overview,
    format_tool_calls,
    sanitize_payload,
    truncate_text,
)
from self_ai.observability import format_trace_payload, register_trace_listener, unregister_trace_listener

DEFAULT_CHAT_TITLE = "Untitled chat"
PROJECT_ROOT = Path(__file__).resolve().parent


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_json(data: Any) -> str:
    return json.dumps(sanitize_payload(data), ensure_ascii=False, indent=2, default=str)


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_chat_id() -> str:
    return "chat-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]


def _make_chat_title(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return DEFAULT_CHAT_TITLE
    return truncate_text(cleaned, 46)


def _create_chat_record(chat_id: str | None = None, *, title: str = DEFAULT_CHAT_TITLE) -> dict[str, Any]:
    now = _now_label()
    return {
        "chat_id": chat_id or _new_chat_id(),
        "title": title or DEFAULT_CHAT_TITLE,
        "created_at": now,
        "updated_at": now,
        "turn_count": 0,
        "runs": [],
    }


def _extract_run_metrics(result: Any, logs: list[str], elapsed_ms: int) -> dict[str, Any]:
    data = _as_dict(result)
    overview = format_run_overview(data, logs=logs)
    runtime = format_runtime_signals(data, logs=logs)
    return {
        "run_id": overview.get("run_id", ""),
        "model": overview.get("model", ""),
        "elapsed_ms": int(elapsed_ms or overview.get("elapsed_ms", 0) or 0),
        "error_count": int(overview.get("error_count", 0) or 0),
        "terminal_error_count": int(runtime.get("terminal_error_count", 0) or 0),
        "control_event_count": int(runtime.get("control_event_count", 0) or 0),
        "fusion_failed": bool(runtime.get("fusion_failed", False)),
        "degraded": bool(runtime.get("degraded", False)),
        "memory_committed_count": int(runtime.get("memory_committed_count", 0) or 0),
        "model_call_count": int(runtime.get("model_call_count", 0) or 0),
        "tool_call_count": int(runtime.get("tool_call_count", 0) or 0),
    }


def _append_chat_run(
    chat: dict[str, Any],
    *,
    question: str,
    result: Any,
    error: dict[str, Any] | None,
    logs: list[str],
    elapsed_ms: int,
) -> dict[str, Any]:
    runs = chat.setdefault("runs", [])
    if not isinstance(runs, list):
        runs = []
        chat["runs"] = runs
    metrics = _extract_run_metrics(result, logs, elapsed_ms) if error is None else {
        "elapsed_ms": int(elapsed_ms or 0),
        "error_count": 1,
        "terminal_error_count": 0,
        "control_event_count": 0,
        "fusion_failed": False,
        "degraded": False,
        "memory_committed_count": 0,
        "model_call_count": 0,
        "tool_call_count": 0,
    }
    record = {
        "question": question,
        "result": result,
        "error": error,
        "logs": list(logs),
        "elapsed_ms": int(elapsed_ms or 0),
        "created_at": _now_label(),
        "metrics": metrics,
    }
    runs.append(record)
    chat["turn_count"] = len(runs)
    chat["updated_at"] = record["created_at"]
    if chat.get("title") == DEFAULT_CHAT_TITLE:
        chat["title"] = _make_chat_title(question)
    return record


def _append_pending_chat_run(chat: dict[str, Any], *, question: str, pending_token: str) -> int:
    runs = chat.setdefault("runs", [])
    if not isinstance(runs, list):
        runs = []
        chat["runs"] = runs
    record = {
        "question": question,
        "result": None,
        "error": None,
        "logs": [],
        "elapsed_ms": 0,
        "created_at": _now_label(),
        "metrics": {
            "run_id": "",
            "model": "",
            "elapsed_ms": 0,
            "error_count": 0,
            "terminal_error_count": 0,
            "control_event_count": 0,
            "fusion_failed": False,
            "degraded": False,
            "memory_committed_count": 0,
            "model_call_count": 0,
            "tool_call_count": 0,
        },
        "pending": True,
        "pending_token": str(pending_token or ""),
    }
    runs.append(record)
    chat["turn_count"] = len(runs)
    chat["updated_at"] = record["created_at"]
    if chat.get("title") == DEFAULT_CHAT_TITLE:
        chat["title"] = _make_chat_title(question)
    return len(runs) - 1


def _execute_question(
    question: str,
    *,
    session_id: str = "default",
    chat_id: str | None = None,
    on_log: Callable[[str], None] | None = None,
) -> tuple[Any, dict[str, Any] | None, list[str], int]:
    """Execute one workflow call and capture trace logs and elapsed time."""
    logs: list[str] = []
    started = time.perf_counter()

    def emit(line: str) -> None:
        logs.append(line)
        if on_log is not None:
            on_log(line)

    def listener(payload: dict[str, Any]) -> None:
        emit(format_trace_payload(payload))

    register_trace_listener(listener)
    try:
        result = asyncio.run(
            run_autonomy_workflow(
                question,
                session_id=session_id,
                chat_id=chat_id or session_id,
            )
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result, None, logs, elapsed_ms
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return (
            None,
            {
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "error_type": type(exc).__name__,
            },
            logs,
            elapsed_ms,
        )
    finally:
        unregister_trace_listener(listener)


def run_frontend_workflow_once(question: str, *, session_id: str = "default") -> dict[str, Any]:
    """Public helper for manual smoke runs outside Streamlit UI."""
    result, error, logs, elapsed_ms = _execute_question(
        question,
        session_id=session_id,
        chat_id=session_id,
    )
    if isinstance(result, dict):
        result.setdefault("metadata", {})
        if isinstance(result["metadata"], dict):
            result["metadata"].setdefault("elapsed_ms", elapsed_ms)
    return {
        "result": result,
        "error": error,
        "logs": logs,
        "elapsed_ms": elapsed_ms,
    }


def _execute_tool_once(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    session_id: str = "default",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        result = asyncio.run(
            run_tool_once(
                tool_name,
                arguments,
                session_id=session_id,
                metadata={"source": "frontend_ops_panel"},
            )
        )
        return result, None
    except Exception as exc:
        return None, {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }


def _worker(
    question: str,
    session_id: str,
    chat_id: str,
    pending_token: str,
    out_queue: "queue.Queue[dict[str, Any]]",
) -> None:
    def push_log(line: str) -> None:
        out_queue.put(
            {
                "type": "log",
                "line": line,
                "chat_id": chat_id,
                "pending_token": pending_token,
                "question": question,
            }
        )

    result, error, _logs, elapsed_ms = _execute_question(
        question,
        session_id=session_id,
        chat_id=chat_id,
        on_log=push_log,
    )
    if error is not None:
        out_queue.put(
            {
                "type": "error",
                **error,
                "elapsed_ms": elapsed_ms,
                "chat_id": chat_id,
                "pending_token": pending_token,
                "question": question,
            }
        )
    else:
        if isinstance(result, dict):
            result.setdefault("metadata", {})
            if isinstance(result["metadata"], dict):
                result["metadata"].setdefault("elapsed_ms", elapsed_ms)
        out_queue.put(
            {
                "type": "result",
                "data": result,
                "elapsed_ms": elapsed_ms,
                "chat_id": chat_id,
                "pending_token": pending_token,
                "question": question,
            }
        )
    out_queue.put(
        {
            "type": "done",
            "chat_id": chat_id,
            "pending_token": pending_token,
            "question": question,
        }
    )


def _init_state() -> None:
    if "chats" not in st.session_state:
        first_chat = _create_chat_record("default", title="Default chat")
        st.session_state.chats = {"default": first_chat}
    st.session_state.setdefault("active_chat_id", "default")
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("thread", None)
    st.session_state.setdefault("queue", None)
    st.session_state.setdefault("elapsed_ms", 0)
    st.session_state.setdefault("session_id", "default")
    st.session_state.setdefault("pending_question", "")
    st.session_state.setdefault("active_page", "chat")


def _get_active_chat() -> dict[str, Any]:
    chats = st.session_state.setdefault("chats", {})
    if not isinstance(chats, dict):
        chats = {}
        st.session_state.chats = chats
    active = str(st.session_state.get("active_chat_id", "default") or "default")
    if active not in chats:
        chats[active] = _create_chat_record(active)
    return chats[active]


def _load_chat_snapshot(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id
    st.session_state.session_id = chat_id
    chat = _get_active_chat()
    runs = chat.get("runs", [])
    latest = runs[-1] if isinstance(runs, list) and runs else {}
    st.session_state.logs = list(latest.get("logs", [])) if isinstance(latest, dict) else []
    st.session_state.result = latest.get("result") if isinstance(latest, dict) else None
    st.session_state.error = latest.get("error") if isinstance(latest, dict) else None
    st.session_state.elapsed_ms = int(latest.get("elapsed_ms", 0) or 0) if isinstance(latest, dict) else 0


def _create_new_chat() -> str:
    chat = _create_chat_record()
    st.session_state.chats[chat["chat_id"]] = chat
    _load_chat_snapshot(chat["chat_id"])
    st.session_state.question = ""
    return chat["chat_id"]


def _start_run(question: str, *, session_id: str) -> None:
    chat = _get_active_chat()
    pending_token = f"{chat.get('chat_id', 'default')}::{datetime.now().timestamp()}::{uuid4().hex[:10]}"
    _append_pending_chat_run(chat, question=question, pending_token=pending_token)
    out_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    thread = threading.Thread(
        target=_worker,
        args=(question, session_id, session_id, pending_token, out_queue),
        daemon=True,
    )
    st.session_state.running = True
    st.session_state.logs = []
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.elapsed_ms = 0
    st.session_state.thread = thread
    st.session_state.queue = out_queue
    st.session_state.pending_question = question
    thread.start()


def _drain_queue() -> None:
    out_queue = st.session_state.queue
    if out_queue is None:
        return
    while True:
        try:
            item = out_queue.get_nowait()
        except queue.Empty:
            break
        item_type = item.get("type")
        item_chat_id = str(item.get("chat_id", "") or "")
        item_pending_token = str(item.get("pending_token", "") or "")
        item_question = str(item.get("question", "") or "")

        def _find_or_create_target_run() -> tuple[dict[str, Any], list[dict[str, Any]], int]:
            chat = _get_active_chat()
            if item_chat_id and str(chat.get("chat_id", "")) != item_chat_id:
                chats = st.session_state.setdefault("chats", {})
                if isinstance(chats, dict):
                    if item_chat_id not in chats:
                        chats[item_chat_id] = _create_chat_record(item_chat_id)
                    chat = chats[item_chat_id] if isinstance(chats[item_chat_id], dict) else chat
            runs = chat.get("runs", [])
            if not isinstance(runs, list):
                runs = []
                chat["runs"] = runs
            for idx, rec in enumerate(runs):
                if isinstance(rec, dict) and str(rec.get("pending_token", "") or "") == item_pending_token:
                    return chat, runs, idx
            # fallback: avoid duplicate user turns by finding same question pending
            for idx, rec in enumerate(runs):
                if (
                    isinstance(rec, dict)
                    and bool(rec.get("pending", False))
                    and str(rec.get("question", "") or "") == item_question
                ):
                    return chat, runs, idx
            new_idx = _append_pending_chat_run(chat, question=item_question, pending_token=item_pending_token)
            runs = chat.get("runs", []) if isinstance(chat.get("runs"), list) else []
            return chat, runs, new_idx

        if item_type == "log":
            st.session_state.logs.append(item.get("line", ""))
        elif item_type == "result":
            st.session_state.result = item.get("data")
            st.session_state.elapsed_ms = int(item.get("elapsed_ms", 0) or 0)
            chat, runs, idx = _find_or_create_target_run()
            if 0 <= idx < len(runs) and isinstance(runs[idx], dict):
                runs[idx]["result"] = st.session_state.result
                runs[idx]["error"] = None
                runs[idx]["logs"] = list(st.session_state.logs)
                runs[idx]["elapsed_ms"] = st.session_state.elapsed_ms
                runs[idx]["metrics"] = _extract_run_metrics(
                    st.session_state.result,
                    st.session_state.logs,
                    st.session_state.elapsed_ms,
                )
                runs[idx]["pending"] = False
                chat["updated_at"] = _now_label()
        elif item_type == "error":
            st.session_state.error = {
                "message": item.get("message", "Unknown error"),
                "traceback": item.get("traceback", ""),
                "error_type": item.get("error_type", "Exception"),
            }
            st.session_state.elapsed_ms = int(item.get("elapsed_ms", 0) or 0)
            chat, runs, idx = _find_or_create_target_run()
            if 0 <= idx < len(runs) and isinstance(runs[idx], dict):
                runs[idx]["result"] = None
                runs[idx]["error"] = dict(st.session_state.error)
                runs[idx]["logs"] = list(st.session_state.logs)
                runs[idx]["elapsed_ms"] = st.session_state.elapsed_ms
                runs[idx]["metrics"] = {
                    "run_id": "",
                    "model": "",
                    "elapsed_ms": st.session_state.elapsed_ms,
                    "error_count": 1,
                    "terminal_error_count": 0,
                    "control_event_count": 0,
                    "fusion_failed": False,
                    "degraded": False,
                    "memory_committed_count": 0,
                    "model_call_count": 0,
                    "tool_call_count": 0,
                }
                runs[idx]["pending"] = False
                chat["updated_at"] = _now_label()
        elif item_type == "done":
            st.session_state.running = False


def _inject_design() -> None:
    st.markdown(
        """
        <style>
        :root {
          --sai-ink: #14213d;
          --sai-muted: #687684;
          --sai-rust: #bb4d2f;
          --sai-red: #ff4b4b;
          --sai-sand: #f5efe1;
          --sai-mint: #2a9d8f;
          --sai-slate: #22333b;
          --sai-night: #0f141e;
        }
        .stApp {
          background:
            radial-gradient(circle at 8% 8%, rgba(42,157,143,.18), transparent 28%),
            radial-gradient(circle at 92% 0%, rgba(187,77,47,.16), transparent 24%),
            linear-gradient(135deg, #fbf7ed 0%, #eef3ef 48%, #f7efe6 100%);
          color: var(--sai-ink);
        }
        .stApp, .stApp p, .stApp label {
          color: var(--sai-ink);
        }
        .stApp p, .stApp label {
          font-weight: 500;
        }
        h1, h2, h3 {
          font-family: "Georgia", "Cambria", "Palatino Linotype", serif;
          letter-spacing: -0.03em;
          color: var(--sai-ink) !important;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #13262f 0%, #22333b 100%);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
          color: #fff7e8 !important;
        }
        [data-testid="stSidebar"] input {
          background: #0b121b !important;
          color: #fff7e8 !important;
          -webkit-text-fill-color: #fff7e8 !important;
          border: 1px solid rgba(255,247,232,.18) !important;
        }
        [data-testid="stSidebar"] input:disabled,
        [data-testid="stSidebar"] input[disabled] {
          background: rgba(255,247,232,.95) !important;
          color: #13262f !important;
          -webkit-text-fill-color: #13262f !important;
          opacity: 1 !important;
          border: 1px solid rgba(255,247,232,.55) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] span {
          color: #fff7e8 !important;
        }
        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input {
          background: rgba(255, 250, 240, .96) !important;
          color: var(--sai-ink) !important;
          border: 1px solid rgba(20,33,61,.22) !important;
          border-radius: 14px !important;
          box-shadow: inset 0 1px 0 rgba(255,255,255,.7);
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
          color: #73808c !important;
          opacity: 1 !important;
        }
        .stButton > button,
        .stDownloadButton > button {
          border-radius: 12px !important;
          border: 1px solid rgba(20,33,61,.16) !important;
          background: #121822 !important;
          color: #fff7e8 !important;
          font-weight: 700 !important;
          box-shadow: 0 8px 20px rgba(20,33,61,.10);
        }
        .stButton > button p,
        .stDownloadButton > button p,
        .stButton > button span,
        .stDownloadButton > button span {
          color: #fff7e8 !important;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
          background: var(--sai-red) !important;
          border-color: rgba(255,75,75,.85) !important;
          color: #fff7e8 !important;
        }
        .stButton > button:disabled,
        .stDownloadButton > button:disabled {
          background: rgba(18,24,34,.42) !important;
          color: rgba(255,247,232,.72) !important;
          opacity: 1 !important;
        }
        .stButton > button:disabled p,
        .stDownloadButton > button:disabled p {
          color: rgba(255,247,232,.72) !important;
        }
        button[data-baseweb="tab"] p {
          color: var(--sai-muted) !important;
          font-weight: 700 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] p {
          color: var(--sai-red) !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span {
          color: #71808e !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] span {
          color: rgba(255,247,232,.78) !important;
        }
        [data-testid="stCodeBlock"],
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code,
        [data-testid="stCodeBlock"] span,
        pre,
        code {
          color: #f8fafc !important;
        }
        [data-testid="stCodeBlock"] {
          background: #171b24 !important;
          border: 1px solid rgba(255,255,255,.08) !important;
          border-radius: 14px !important;
        }
        div[data-testid="stMetric"] {
          background: rgba(255,255,255,.72);
          border: 1px solid rgba(20,33,61,.10);
          border-radius: 18px;
          padding: 14px 16px;
          box-shadow: 0 12px 34px rgba(34,51,59,.08);
        }
        .sai-hero {
          border: 1px solid rgba(20,33,61,.10);
          border-radius: 26px;
          padding: 24px 28px;
          background: linear-gradient(135deg, rgba(255,255,255,.78), rgba(245,239,225,.64));
          box-shadow: 0 18px 48px rgba(34,51,59,.10);
          margin-bottom: 18px;
        }
        .sai-hero strong { color: var(--sai-rust); }
        .sai-titlebar {
          border: 1px solid rgba(20,33,61,.10);
          border-radius: 16px;
          padding: 12px 14px;
          background: linear-gradient(135deg, rgba(255,255,255,.80), rgba(245,239,225,.65));
          box-shadow: 0 10px 26px rgba(34,51,59,.08);
          margin-bottom: 12px;
        }
        .sai-titlebar h2 {
          margin: 0;
          font-size: 1.4rem;
        }
        .sai-titlebar p {
          margin: 4px 0 0;
          color: #5c6a77;
        }
        .sai-chat-window {
          height: 62vh;
          overflow-y: auto;
          border: 1px solid rgba(20,33,61,.12);
          border-radius: 18px;
          background: rgba(255,255,255,.55);
          padding: 10px 14px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,.85);
          margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chat_labels() -> list[str]:
    chats = st.session_state.get("chats", {})
    if not isinstance(chats, dict):
        return ["default"]
    ordered = sorted(
        chats.values(),
        key=lambda item: str(_as_dict(item).get("updated_at", "")),
        reverse=True,
    )
    return [
        f"{_as_dict(item).get('title', DEFAULT_CHAT_TITLE)} · {_as_dict(item).get('chat_id', '')}"
        for item in ordered
    ]


def _chat_id_from_label(label: str) -> str:
    if " · " in label:
        return label.rsplit(" · ", 1)[-1]
    return label


def _load_latest_benchmark_snapshot(root: str | Path = PROJECT_ROOT / "benchmarks" / "runs") -> dict[str, Any]:
    """Load the newest benchmark run for a lightweight project-health panel."""
    runs_root = Path(root)
    if not runs_root.exists():
        return {"run_dir": "", "metrics": {}, "cases": [], "failures": [], "error": ""}

    run_dirs = [item for item in runs_root.iterdir() if item.is_dir()]
    if not run_dirs:
        return {"run_dir": "", "metrics": {}, "cases": [], "failures": [], "error": ""}

    latest = max(run_dirs, key=lambda item: (item.stat().st_mtime, item.name))
    snapshot: dict[str, Any] = {"run_dir": str(latest), "metrics": {}, "cases": [], "failures": [], "error": ""}
    try:
        metrics_path = latest / "metrics.json"
        cases_path = latest / "case_results.json"
        failures_path = latest / "failures.json"
        if metrics_path.exists():
            snapshot["metrics"] = _as_dict(_read_json_file(metrics_path))
        if cases_path.exists():
            cases = _read_json_file(cases_path)
            snapshot["cases"] = cases if isinstance(cases, list) else []
        if failures_path.exists():
            failures = _read_json_file(failures_path)
            snapshot["failures"] = failures if isinstance(failures, list) else []
    except Exception as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return sanitize_payload(snapshot)


def _render_chat_sidebar() -> None:
    st.header("Self AI")
    st.caption("Chats are isolated by chat_id/session_id.")
    p1, p2 = st.columns(2)
    if p1.button("Chat", width="stretch"):
        st.session_state.active_page = "chat"
        st.rerun()
    if p2.button("Developer", width="stretch"):
        st.session_state.active_page = "developer"
        st.rerun()

    if st.button("New Chat", type="primary", width="stretch"):
        _create_new_chat()
        st.session_state.active_page = "chat"
        st.rerun()

    chat_labels = _chat_labels()
    active = str(st.session_state.get("active_chat_id", "default"))
    active_label = next((label for label in chat_labels if _chat_id_from_label(label) == active), chat_labels[0])
    selected = st.radio(
        "Chats",
        options=chat_labels,
        index=chat_labels.index(active_label),
        disabled=False,
        label_visibility="collapsed",
    )
    selected_chat_id = _chat_id_from_label(selected)
    if selected_chat_id != active:
        _load_chat_snapshot(selected_chat_id)
        st.rerun()

    chat = _get_active_chat()
    st.text_input("Active chat_id", value=chat["chat_id"], disabled=True)
    st.caption(f"Turns: {chat.get('turn_count', 0)} · Updated: {chat.get('updated_at', '')}")


def _render_overview(overview: dict[str, Any]) -> None:
    st.subheader("Run Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Run ID", truncate_text(overview.get("run_id", ""), 28))
    c2.metric("Session ID", truncate_text(overview.get("session_id", ""), 24))
    c3.metric("Execution Mode", overview.get("execution_mode", ""))
    c4.metric("Pipeline", truncate_text(overview.get("pipeline_name", ""), 26))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Agent Outputs", str(overview.get("agent_output_count", 0)))
    c6.metric("Revision Count", str(overview.get("revision_count", 0)))
    c7.metric("Quality Gate", str(overview.get("quality_gate.decision", "")))
    c8.metric("Elapsed (ms)", str(overview.get("elapsed_ms", 0)))


def _render_runtime_cards(result: dict[str, Any], logs: list[str]) -> None:
    runtime = format_runtime_signals(result, logs=logs)
    st.subheader("Runtime Signals")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model Calls", runtime.get("model_call_count", 0))
    c2.metric("Tool Calls", runtime.get("tool_call_count", 0))
    c3.metric("Control Events", runtime.get("control_event_count", 0))
    c4.metric("Terminal Errors", runtime.get("terminal_error_count", 0))
    c5.metric("Committed Memories", runtime.get("memory_committed_count", 0))

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Fusion Failed", str(runtime.get("fusion_failed", False)))
    c7.metric("Degraded", str(runtime.get("degraded", False)))
    c8.metric("Recent Turns", f"{runtime.get('recent_turns_kept', 0)}/{runtime.get('recent_turns_requested', 0)}")
    c9.metric("Prompt Chars", runtime.get("prompt_chars_total", 0))

    with st.expander("Memory and Layer Health", expanded=False):
        st.write(
            {
                "layer_health": runtime.get("layer_health", {}),
                "degraded_reasons": runtime.get("degraded_reasons", []),
                "issue_write_allowed": runtime.get("issue_write_allowed", False),
                "issue_suppressed_due_to_terminal_success": runtime.get(
                    "issue_suppressed_due_to_terminal_success",
                    False,
                ),
                "sidecar": {
                    "attempted": runtime.get("sidecar_attempted", False),
                    "succeeded": runtime.get("sidecar_succeeded", False),
                    "fallback_used": runtime.get("sidecar_fallback_used", False),
                },
                "l2l3": {
                    "failed": runtime.get("l2l3_failed", False),
                    "qdrant_ok": runtime.get("qdrant_ok", 0),
                    "neo4j_ok": runtime.get("neo4j_ok", 0),
                },
            }
        )


def _render_chat_history() -> None:
    chat = _get_active_chat()
    runs = chat.get("runs", [])
    if not isinstance(runs, list) or not runs:
        st.info("This chat has no completed turns yet.")
        return
    rows = []
    for idx, run in enumerate(runs, start=1):
        item = _as_dict(run)
        metrics = _as_dict(item.get("metrics"))
        rows.append(
            {
                "turn": idx,
                "time": item.get("created_at", ""),
                "task": truncate_text(item.get("question", ""), 90),
                "model": metrics.get("model", ""),
                "elapsed_ms": metrics.get("elapsed_ms", 0),
                "errors": metrics.get("error_count", 0),
                "memories": metrics.get("memory_committed_count", 0),
                "fusion_failed": metrics.get("fusion_failed", False),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_chat_transcript(*, show_agent_details: bool, max_display_chars: int) -> None:
    chat = _get_active_chat()
    runs = chat.get("runs", [])
    if not isinstance(runs, list) or not runs:
        st.info("Start a new message to begin this isolated chat.")
        return

    chat_box = st.container(height=640)
    for idx, run in enumerate(runs, start=1):
        item = _as_dict(run)
        question = str(item.get("question", "") or "").strip()
        result = _as_dict(item.get("result"))
        err = _as_dict(item.get("error"))
        metrics = _as_dict(item.get("metrics"))
        created_at = str(item.get("created_at", "") or "")
        final = format_final_response(result)
        assistant_text = "Running... generating response."
        if not bool(item.get("pending", False)):
            if err:
                assistant_text = str(err.get("message", "Unknown error") or "Unknown error")
            else:
                assistant_text = str(final.get("response", "") or "")
        with chat_box:
            st.markdown(
                (
                    "<div style='display:flex;justify-content:flex-end;margin:8px 0;'>"
                    f"<div style='max-width:78%;background:#ff4b4b;color:#fff8ee;border:1px solid rgba(255,75,75,.8);"
                    "border-radius:18px;padding:12px 14px;line-height:1.52;box-shadow:0 6px 18px rgba(20,33,61,.06);"
                    "word-break:break-word;'>"
                    f"{html.escape(truncate_text(question or '(empty message)', max(500, max_display_chars * 4))).replace(chr(10), '<br/>')}"
                    "</div></div>"
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                (
                    "<div style='display:flex;justify-content:flex-start;margin:8px 0;'>"
                    "<div style='max-width:78%;background:rgba(255,255,255,.9);color:#14213d;border:1px solid rgba(20,33,61,.09);"
                    "border-radius:18px;padding:12px 14px;line-height:1.52;box-shadow:0 6px 18px rgba(20,33,61,.06);"
                    "word-break:break-word;'>"
                    f"{html.escape(truncate_text(assistant_text, max(500, max_display_chars * 4))).replace(chr(10), '<br/>')}"
                    "</div></div>"
                ),
                unsafe_allow_html=True,
            )
            meta_text = (
                f"Turn {idx} · {created_at}"
                if not show_agent_details
                else f"Turn {idx} · {created_at} · {metrics.get('model', '')} · {metrics.get('elapsed_ms', 0)}ms"
            )
            st.caption(meta_text)


def _render_developer_page(
    *,
    show_trace: bool,
    show_tool_calls: bool,
    show_agent_details: bool,
    show_raw_json: bool,
    max_display_chars: int,
    show_ops_panel: bool,
) -> None:
    st.subheader("Developer")
    history_tab, result_tab, trace_tab, health_tab = st.tabs(
        ["Chat History", "Current Result", "Live Trace", "Project Health"]
    )
    with history_tab:
        _render_chat_history()

    with trace_tab:
        st.subheader("Live Trace Logs")
        st.code("\n".join(st.session_state.logs[-500:]) if st.session_state.logs else "(waiting)", language="json")

    with health_tab:
        _render_benchmark_panel()

    with result_tab:
        st.subheader("Result Panel")
        if st.session_state.error:
            err = _as_dict(st.session_state.error)
            st.error(err.get("message", "Unknown error"))
            st.write({"error_type": err.get("error_type", "Exception"), "elapsed_ms": st.session_state.elapsed_ms})
            with st.expander("Exception Traceback", expanded=False):
                st.code(str(err.get("traceback", "")), language="text")
        elif st.session_state.result is not None:
            result = _as_dict(st.session_state.result)
            result.setdefault("metadata", {})
            if isinstance(result["metadata"], dict):
                result["metadata"].setdefault("elapsed_ms", st.session_state.elapsed_ms)
            _render_sections(
                result,
                st.session_state.logs,
                show_trace=show_trace,
                show_tool_calls=show_tool_calls,
                show_agent_details=show_agent_details,
                show_raw_json=show_raw_json,
                max_display_chars=int(max_display_chars),
            )
        else:
            st.info("Waiting for result...")

    if show_ops_panel:
        _render_workspace_ops(st.session_state.active_chat_id)


def _render_benchmark_panel() -> None:
    snapshot = _load_latest_benchmark_snapshot()
    if snapshot.get("error"):
        st.error("Could not load latest benchmark snapshot: " + str(snapshot.get("error")))
        return
    if not snapshot.get("run_dir"):
        st.info("No benchmark run found under benchmarks/runs.")
        return

    metrics = _as_dict(snapshot.get("metrics"))
    cases = snapshot.get("cases", [])
    failures = snapshot.get("failures", [])
    st.subheader("Latest Benchmark")
    st.caption(str(snapshot.get("run_dir", "")))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verified Success", metrics.get("verified_success_rate", 0))
    c2.metric("False Success", metrics.get("false_success_rate", 0))
    c3.metric("Failure Count", metrics.get("failure_count", 0))
    c4.metric("Complex Code Pytest", metrics.get("complex_code_pytest_pass_rate", 0))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Terminal Error Rate", metrics.get("terminal_error_rate", 0))
    c6.metric("Issue Write on Success", metrics.get("issue_write_when_success_rate", 0))
    c7.metric("Fusion Hard Failed", metrics.get("fusion_hard_failed_rate", 0))
    c8.metric("Avg Elapsed ms", metrics.get("avg_elapsed_ms", 0))

    case_rows = []
    if isinstance(cases, list):
        for item in cases:
            case = _as_dict(item)
            common = _as_dict(case.get("common"))
            case_rows.append(
                {
                    "case_id": case.get("case_id", ""),
                    "ok": bool(case.get("validation_ok", False)),
                    "elapsed_ms": case.get("elapsed_ms", 0),
                    "model_calls": common.get("model_calls", 0),
                    "tool_calls": common.get("tool_calls", 0),
                    "terminal_errors": common.get("terminal_error_count", 0),
                    "degraded": common.get("degraded", False),
                }
            )
    if case_rows:
        st.dataframe(case_rows, width="stretch", hide_index=True)

    if isinstance(failures, list) and failures:
        with st.expander("Historical / Recorded Failures", expanded=False):
            st.dataframe(
                [
                    {
                        "case_id": _as_dict(item).get("case_id", ""),
                        "severity": _as_dict(item).get("severity", ""),
                        "failure_type": _as_dict(item).get("failure_type", ""),
                        "reason": truncate_text(_as_dict(item).get("reason", ""), 120),
                    }
                    for item in failures
                ],
                width="stretch",
                hide_index=True,
            )


def _render_sections(
    result_raw: Any,
    logs: list[str],
    *,
    show_trace: bool,
    show_tool_calls: bool,
    show_agent_details: bool,
    show_raw_json: bool,
    max_display_chars: int,
) -> None:
    result = _as_dict(result_raw)

    overview = format_run_overview(result, logs=logs)
    timeline = format_engine_timeline(result, logs)
    tool_calls = format_tool_calls(result, logs)
    agent_turns = format_agent_turns(result)
    revision = format_revision_summary(result, logs=logs)
    reviews = format_review_reports(result)
    gate = format_quality_gate(result)
    memory = format_memory_context(result)
    final = format_final_response(result)
    errors = format_errors(result, logs=logs)

    _render_overview(overview)
    _render_runtime_cards(result, logs)

    diagnostics_tab, memory_tab, agents_tab, raw_tab = st.tabs(
        ["Diagnostics", "Memory", "Agents & Review", "Exports"]
    )

    with diagnostics_tab:
        if show_trace:
            st.subheader("Engine Timeline")
            st.write(timeline)

        if show_tool_calls:
            st.subheader("Tool Calls")
            st.write(tool_calls)

        st.subheader("Errors")
        st.write(errors)

    with agents_tab:
        st.subheader("Agent Turns")
        st.write({"selected_agents": agent_turns.get("selected_agents", [])})
        if show_agent_details:
            st.write(agent_turns.get("turns", []))

        st.subheader("Revision Loop")
        st.write(revision)

        st.subheader("Review Reports")
        st.write(reviews)

        st.subheader("Quality Gate")
        st.write(gate)

    with memory_tab:
        st.subheader("Memory / Context")
        st.write({
            "evidence_count": memory.get("evidence_count", 0),
            "source_type_counts": memory.get("source_type_counts", {}),
            "context_preview": truncate_text(memory.get("context_preview", ""), max_display_chars),
            "graph_path_count": memory.get("graph_path_count", 0),
            "graph_relation_type_counts": memory.get("graph_relation_type_counts", {}),
            "graph_context_preview": truncate_text(memory.get("graph_context_preview", ""), max_display_chars),
            "token_budget": memory.get("token_budget", 0),
        })

    st.subheader("Final Response")
    st.text_area("Response", value=truncate_text(final.get("response", ""), max(500, max_display_chars * 4)), height=220, disabled=True)
    st.caption("model: " + str(final.get("model", "")))

    result_json = _safe_json(result_raw)
    logs_json = _safe_json(logs)
    frontend_session = _safe_json({"overview": overview, "timeline_count": len(timeline), "tool_call_count": len(tool_calls)})

    with raw_tab:
        st.subheader("Raw JSON Export")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download run_result.json", data=result_json.encode("utf-8"), file_name="run_result.json", mime="application/json", width="stretch")
        c2.download_button("Download trace_logs.json", data=logs_json.encode("utf-8"), file_name="trace_logs.json", mime="application/json", width="stretch")
        c3.download_button("Download frontend_session.json", data=frontend_session.encode("utf-8"), file_name="frontend_session.json", mime="application/json", width="stretch")
        if show_raw_json:
            with st.expander("Raw Result JSON", expanded=False):
                st.code(result_json, language="json")


def _render_workspace_ops(session_id: str) -> None:
    st.subheader("Workspace Operations")
    st.caption("Real operations through ToolRuntime permissions.")

    with st.expander("File List", expanded=False):
        list_path = st.text_input("Path", value=".", key="ops_list_path")
        list_recursive = st.checkbox("Recursive", value=False, key="ops_list_recursive")
        list_max_entries = st.number_input("Max entries", min_value=1, max_value=2000, value=200, key="ops_list_max")
        if st.button("Run workspace.file.list", key="ops_list_btn"):
            result, error = _execute_tool_once("workspace.file.list", {"path": list_path, "recursive": list_recursive, "max_entries": int(list_max_entries)}, session_id=session_id)
            if error:
                st.error(error["message"])
            else:
                st.json(sanitize_payload(result))

    with st.expander("File Read", expanded=False):
        read_path = st.text_input("Read path", value="README.md", key="ops_read_path")
        read_max_chars = st.number_input("Max chars", min_value=100, max_value=200000, value=50000, step=100, key="ops_read_max_chars")
        if st.button("Run workspace.file.read", key="ops_read_btn"):
            result, error = _execute_tool_once("workspace.file.read", {"path": read_path, "max_chars": int(read_max_chars)}, session_id=session_id)
            if error:
                st.error(error["message"])
            else:
                st.json(sanitize_payload(result))

    with st.expander("File Write / Edit", expanded=False):
        st.warning("Real writes under project_root. Confirm before execution.")
        write_path = st.text_input("Write path", value="artifacts/ui_write_sample.txt", key="ops_write_path")
        write_content = st.text_area("Write content", value="hello from cockpit", height=100, key="ops_write_content")
        if st.checkbox("Confirm write", value=False, key="ops_write_confirm") and st.button("Run workspace.file.write", key="ops_write_btn"):
            result, error = _execute_tool_once("workspace.file.write", {"path": write_path, "content": write_content}, session_id=session_id)
            if error:
                st.error(error["message"])
            else:
                st.json(sanitize_payload(result))

        edit_path = st.text_input("Edit path", value="artifacts/ui_write_sample.txt", key="ops_edit_path")
        old_text = st.text_area("Old text", value="hello", height=70, key="ops_edit_old")
        new_text = st.text_area("New text", value="hello updated", height=70, key="ops_edit_new")
        if st.checkbox("Confirm edit", value=False, key="ops_edit_confirm") and st.button("Run workspace.file.edit", key="ops_edit_btn"):
            result, error = _execute_tool_once("workspace.file.edit", {"path": edit_path, "old_text": old_text, "new_text": new_text}, session_id=session_id)
            if error:
                st.error(error["message"])
            else:
                st.json(sanitize_payload(result))

    with st.expander("Shell Exec (guarded)", expanded=False):
        st.info("Requires dev_full + shell enabled + allowlist/denylist checks.")
        shell_cmd = st.text_input("Command", value="git status", key="ops_shell_cmd")
        shell_cwd = st.text_input("CWD", value=".", key="ops_shell_cwd")
        shell_timeout = st.number_input("Timeout (s)", min_value=1, max_value=120, value=20, key="ops_shell_timeout")
        if st.button("Run workspace.shell.exec", key="ops_shell_btn"):
            result, error = _execute_tool_once("workspace.shell.exec", {"command": shell_cmd, "cwd": shell_cwd, "timeout_s": int(shell_timeout)}, session_id=session_id)
            if error:
                st.error(error["message"])
            else:
                st.json(sanitize_payload(result))


def main() -> None:
    st.set_page_config(page_title="Self AI Agent Cockpit", layout="wide")
    _init_state()
    _inject_design()

    with st.sidebar:
        _render_chat_sidebar()
        st.divider()
        st.subheader("View")
        show_trace = st.toggle("Show trace", value=True)
        show_tool_calls = st.toggle("Show tool calls", value=True)
        show_raw_json = st.toggle("Show raw JSON", value=False)
        show_agent_details = st.toggle("Show agent details", value=True)
        show_ops_panel = st.toggle("Show workspace ops panel", value=True)
        max_display_chars = st.number_input("Max display chars", min_value=200, max_value=5000, value=1200, step=100)
        st.info("Safe mode: sensitive payload fields are redacted; shell/file writes remain permission-guarded.")

    st.markdown(
        """
        <div class="sai-titlebar">
          <h2>Self AI Chat</h2>
          <p>Isolated chat execution powered by Kernel -> EngineLoop -> ToolRuntime.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_chat = _get_active_chat()
    st.caption(f"Active isolated chat: `{active_chat['chat_id']}`")

    _drain_queue()

    active_page = str(st.session_state.get("active_page", "chat") or "chat")
    if active_page == "developer":
        _render_developer_page(
            show_trace=show_trace,
            show_tool_calls=show_tool_calls,
            show_agent_details=show_agent_details,
            show_raw_json=show_raw_json,
            max_display_chars=int(max_display_chars),
            show_ops_panel=show_ops_panel,
        )
    else:
        _render_chat_transcript(
            show_agent_details=show_agent_details,
            max_display_chars=int(max_display_chars),
        )
        prompt = st.chat_input(
            "Send a message to this isolated chat...",
            disabled=st.session_state.running,
        )
        if prompt is not None and str(prompt).strip():
            _start_run(
                str(prompt).strip(),
                session_id=str(st.session_state.active_chat_id or "default"),
            )
            st.rerun()

    if st.session_state.running:
        st.status("Running...", state="running", expanded=False)
        time.sleep(0.2)
        st.rerun()
    elif st.session_state.result is not None:
        st.status("Completed", state="complete", expanded=False)


if __name__ == "__main__":
    main()
