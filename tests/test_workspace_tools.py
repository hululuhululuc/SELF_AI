# coding=utf-8
"""Tests for workspace tool safety constraints."""

from pathlib import Path
from unittest.mock import patch

import pytest

from self_ai.kernel.run_context import RunContext
from self_ai.runtime.permission_guard import PermissionGuard
from self_ai.runtime.tool_runtime import ToolRuntime


def _runtime_for_workspace(
    root: Path,
    *,
    shell_enabled: bool = False,
    allow_shell_permission: bool = False,
    allow_workspace_write: bool = False,
    max_file_bytes: int = 1_048_576,
) -> ToolRuntime:
    runtime = ToolRuntime(
        permission_guard=PermissionGuard(
            permission_profile={
                "shell_exec": allow_shell_permission,
                "workspace_write": allow_workspace_write,
            }
        ),
        dependencies={
            "project_root": root,
            "shell_enabled": shell_enabled,
            "max_file_bytes": max_file_bytes,
            "web_search_func": lambda _q, _k: [],
        },
    )
    runtime.register_default_tools()
    return runtime


@pytest.mark.asyncio
async def test_workspace_file_read_inside_project_root_success(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.read",
        {"path": "notes.txt"},
        run_context=ctx,
    )
    assert result.ok is True
    assert "hello world" in result.data["content"]
    assert isinstance(result.data.get("mtime_ns"), int)
    assert isinstance(result.data.get("size_bytes"), int)
    assert isinstance(result.data.get("content_hash"), str)
    assert bool(result.data.get("content_hash"))
    assert isinstance(result.data.get("snapshot_id"), str)
    assert bool(result.data.get("snapshot_id"))
    assert isinstance(result.data.get("snapshot_version"), int)


@pytest.mark.asyncio
async def test_workspace_file_list_returns_safe_entries(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "secrets.key").write_text("x", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.list",
        {"path": ".", "max_entries": 50},
        run_context=ctx,
    )
    assert result.ok is True
    entries = result.data.get("entries", [])
    paths = {item.get("path") for item in entries if isinstance(item, dict)}
    assert "a.py" in paths
    assert "b.txt" in paths
    assert ".env" not in paths
    assert "secrets.key" not in paths
    assert isinstance(result.data.get("snapshot_id"), str)
    assert bool(result.data.get("snapshot_id"))
    assert isinstance(result.data.get("snapshot_version"), int)


@pytest.mark.asyncio
async def test_workspace_file_read_path_escape_denied(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.read",
        {"path": "..\\outside.txt"},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative_path",
    [
        ".env",
        "private.key",
        "cert.pem",
        "app.db",
        "data.sqlite",
        "credentials.json",
        "token.txt",
        ".git/config",
        ".venv/pyvenv.cfg",
    ],
)
async def test_workspace_sensitive_paths_denied(tmp_path: Path, relative_path: str) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("secret", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.read",
        {"path": relative_path},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
async def test_workspace_file_read_large_file_denied(tmp_path: Path) -> None:
    file_path = tmp_path / "big.txt"
    file_path.write_text("x" * 120, encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path, max_file_bytes=64)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.read",
        {"path": "big.txt"},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
async def test_workspace_file_write_large_content_denied(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True, max_file_bytes=32)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.write",
        {"path": "out.txt", "content": "a" * 100},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
async def test_workspace_file_edit_result_exceeds_limit_denied(tmp_path: Path) -> None:
    file_path = tmp_path / "edit.txt"
    file_path.write_text("abc", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True, max_file_bytes=4)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.edit",
        {"path": "edit.txt", "old_text": "abc", "new_text": "abcdef"},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
async def test_workspace_small_file_write_and_read_success(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True, max_file_bytes=128)
    ctx = RunContext(project_root=tmp_path)
    write_result = await runtime.execute_by_name(
        "workspace.file.write",
        {"path": "ok.txt", "content": "hello"},
        run_context=ctx,
    )
    assert write_result.ok is True

    read_result = await runtime.execute_by_name(
        "workspace.file.read",
        {"path": "ok.txt"},
        run_context=ctx,
    )
    assert read_result.ok is True
    assert read_result.data["content"] == "hello"


@pytest.mark.asyncio
async def test_workspace_file_create_alias_success(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True, max_file_bytes=128)
    ctx = RunContext(project_root=tmp_path)
    create_result = await runtime.execute_by_name(
        "workspace.file.create",
        {"path": "created.txt", "content": "hello-create"},
        run_context=ctx,
    )
    assert create_result.ok is True
    read_result = await runtime.execute_by_name(
        "workspace.file.read",
        {"path": "created.txt"},
        run_context=ctx,
    )
    assert read_result.ok is True
    assert read_result.data["content"] == "hello-create"


@pytest.mark.asyncio
async def test_workspace_file_rename_success(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    src.write_text("hello", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.rename",
        {"from_path": "src.txt", "to_path": "dst.txt"},
        run_context=ctx,
    )
    assert result.ok is True
    assert not src.exists()
    assert (tmp_path / "dst.txt").exists()


@pytest.mark.asyncio
async def test_workspace_file_rename_alias_keys_success(tmp_path: Path) -> None:
    src = tmp_path / "src_alias.txt"
    src.write_text("hello", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.rename",
        {"source_path": "src_alias.txt", "destination": "dst_alias.txt"},
        run_context=ctx,
    )
    assert result.ok is True
    assert not src.exists()
    assert (tmp_path / "dst_alias.txt").exists()


@pytest.mark.asyncio
async def test_workspace_file_delete_success(tmp_path: Path) -> None:
    p = tmp_path / "dead.txt"
    p.write_text("x", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.delete",
        {"path": "dead.txt"},
        run_context=ctx,
    )
    assert result.ok is True
    assert not p.exists()


@pytest.mark.asyncio
async def test_workspace_file_delete_alias_keys_success(tmp_path: Path) -> None:
    p = tmp_path / "dead_alias.txt"
    p.write_text("x", encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path, allow_workspace_write=True)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.delete",
        {"target_path": "dead_alias.txt"},
        run_context=ctx,
    )
    assert result.ok is True
    assert not p.exists()


@pytest.mark.asyncio
async def test_workspace_output_truncation_still_applies(tmp_path: Path) -> None:
    file_path = tmp_path / "log.txt"
    file_path.write_text("x" * 200, encoding="utf-8")
    runtime = _runtime_for_workspace(tmp_path)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.file.read",
        {"path": "log.txt", "max_chars": 50},
        run_context=ctx,
    )
    assert result.ok is True
    assert len(result.data["content"]) <= 50


@pytest.mark.asyncio
async def test_workspace_shell_exec_default_disabled(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(tmp_path)
    ctx = RunContext(project_root=tmp_path)
    result = await runtime.execute_by_name(
        "workspace.shell.exec",
        {"command": "git status"},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionDenied"


@pytest.mark.asyncio
async def test_workspace_shell_exec_non_whitelist_command_denied(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(
        tmp_path,
        shell_enabled=True,
        allow_shell_permission=True,
    )
    ctx = RunContext(
        project_root=tmp_path,
        permission_profile={"shell_enabled": True, "shell_exec": True},
    )
    result = await runtime.execute_by_name(
        "workspace.shell.exec",
        {"command": "whoami"},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "git status; echo hacked",
        "git status && echo hacked",
    ],
)
async def test_workspace_shell_exec_command_injection_rejected(tmp_path: Path, command: str) -> None:
    runtime = _runtime_for_workspace(
        tmp_path,
        shell_enabled=True,
        allow_shell_permission=True,
    )
    ctx = RunContext(
        project_root=tmp_path,
        permission_profile={"shell_enabled": True, "shell_exec": True},
    )
    result = await runtime.execute_by_name(
        "workspace.shell.exec",
        {"command": command},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["powershell -c dir", "cmd /c dir", "bash -lc ls"])
async def test_workspace_shell_exec_dangerous_shell_binaries_rejected(tmp_path: Path, command: str) -> None:
    runtime = _runtime_for_workspace(
        tmp_path,
        shell_enabled=True,
        allow_shell_permission=True,
    )
    ctx = RunContext(
        project_root=tmp_path,
        permission_profile={"shell_enabled": True, "shell_exec": True},
    )
    result = await runtime.execute_by_name(
        "workspace.shell.exec",
        {"command": command},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
async def test_workspace_shell_exec_cwd_escape_denied(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(
        tmp_path,
        shell_enabled=True,
        allow_shell_permission=True,
    )
    ctx = RunContext(
        project_root=tmp_path,
        permission_profile={"shell_enabled": True, "shell_exec": True},
    )
    result = await runtime.execute_by_name(
        "workspace.shell.exec",
        {"command": "git status", "cwd": ".."},
        run_context=ctx,
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error["type"] == "PermissionError"


@pytest.mark.asyncio
async def test_workspace_shell_exec_whitelisted_command_with_mock_subprocess(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(
        tmp_path,
        shell_enabled=True,
        allow_shell_permission=True,
    )
    ctx = RunContext(
        project_root=tmp_path,
        permission_profile={"shell_enabled": True, "shell_exec": True},
    )

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    with patch("self_ai.adapters.workspace_tools.subprocess.run", return_value=_Proc()) as mocked_run:
        result = await runtime.execute_by_name(
            "workspace.shell.exec",
            {"command": "python -m pytest", "timeout_s": 5},
            run_context=ctx,
        )
    assert result.ok is True
    assert result.data["code"] == 0
    assert result.data["output"] == "ok"
    mocked_run.assert_called_once()


@pytest.mark.asyncio
async def test_workspace_shell_exec_stdout_stderr_truncation(tmp_path: Path) -> None:
    runtime = _runtime_for_workspace(
        tmp_path,
        shell_enabled=True,
        allow_shell_permission=True,
    )
    ctx = RunContext(
        project_root=tmp_path,
        permission_profile={"shell_enabled": True, "shell_exec": True},
    )

    class _Proc:
        returncode = 0
        stdout = "x" * 200
        stderr = "y" * 200

    with patch("self_ai.adapters.workspace_tools.subprocess.run", return_value=_Proc()):
        result = await runtime.execute_by_name(
            "workspace.shell.exec",
            {"command": "git status", "max_output_chars": 50},
            run_context=ctx,
        )
    assert result.ok is True
    assert len(result.data["output"]) <= 50
