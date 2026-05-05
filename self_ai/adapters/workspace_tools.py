# coding=utf-8
"""Workspace file/shell tool adapters with strict safety checks."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..runtime.tool_registry import ToolRegistry
from ..runtime.tool_spec import ToolSpec

ALLOWED_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".swift",
    ".kt",
    ".kts",
    ".php",
    ".rb",
    ".sh",
    ".sql",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".vue",
    ".svelte",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
ALLOWED_NOEXT: set[str] = {"README", "LICENSE", "Makefile"}
SENSITIVE_FILENAMES: set[str] = {".env"}
SENSITIVE_EXTENSIONS: set[str] = {".pem", ".key", ".sqlite", ".db", ".zip", ".exe"}
SENSITIVE_NAME_SUBSTRINGS: set[str] = {
    "credentials",
    "credential",
    "token",
    "secret",
    "password",
}
SENSITIVE_PATH_PARTS: set[str] = {".git", ".venv", "__pycache__"}
DEFAULT_ALLOWED_COMMAND_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("git", "status"),
    ("git", "diff"),
    ("pytest",),
    ("python", "-m", "pytest"),
    ("ruff", "check"),
    ("mypy",),
    ("streamlit", "run"),
)
SHELL_CONTROL_SEQUENCES: tuple[str, ...] = (
    ";",
    "&&",
    "||",
    "|",
    ">",
    ">>",
    "<",
    "`",
    "$(",
    "&",
    "\n",
    "\r",
)
DANGEROUS_COMMAND_TOKENS: set[str] = {
    "rm",
    "del",
    "sudo",
    "shutdown",
    "reboot",
    "curl",
    "wget",
    "powershell",
    "cmd",
    "bash",
    "sh",
}
DEFAULT_MAX_FILE_BYTES = 1_048_576


def _resolve_project_root(
    explicit_root: str | Path | None,
    run_context: Any | None,
) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).resolve()
    if run_context is not None:
        root = getattr(run_context, "project_root", None)
        if root is not None:
            return Path(root).resolve()
    return Path.cwd().resolve()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("path is required")
    path = (project_root / raw_path).resolve()
    if not path.is_relative_to(project_root):
        raise PermissionError("path escapes project_root")
    return path


def _is_sensitive(path: Path) -> bool:
    lower_name = path.name.lower()
    if lower_name in SENSITIVE_FILENAMES:
        return True
    if lower_name.startswith(".env"):
        return True
    if path.suffix.lower() in SENSITIVE_EXTENSIONS:
        return True
    if any(token in lower_name for token in SENSITIVE_NAME_SUBSTRINGS):
        return True
    return any(part in SENSITIVE_PATH_PARTS for part in path.parts)


def _is_allowed_extension(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix:
        return suffix in ALLOWED_EXTENSIONS
    return path.name in ALLOWED_NOEXT


def _truncate_output(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def _get_max_file_bytes(args: dict[str, Any]) -> int:
    max_file_bytes = int(args.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES))
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be > 0")
    if max_file_bytes > 10 * DEFAULT_MAX_FILE_BYTES:
        raise ValueError("max_file_bytes too large")
    return max_file_bytes


def _ensure_file_size(path: Path, max_file_bytes: int) -> None:
    size = int(path.stat().st_size)
    if size > max_file_bytes:
        raise PermissionError(f"file too large: {size} > {max_file_bytes}")


def _ensure_content_size(content: str, max_file_bytes: int) -> None:
    size = len(content.encode("utf-8"))
    if size > max_file_bytes:
        raise PermissionError(f"content too large: {size} > {max_file_bytes}")


def _contains_shell_control(command: str) -> bool:
    return any(token in command for token in SHELL_CONTROL_SEQUENCES)


def _normalize_token(token: str) -> str:
    return token.strip().strip("\"'").lower()


def _split_command(command: str) -> list[str]:
    try:
        argv = shlex.split(command, posix=False)
    except ValueError as exc:
        raise PermissionError(f"invalid command syntax: {exc}") from exc
    if not argv:
        raise ValueError("shell.exec requires command")
    return argv


def _is_allowed_command(argv: list[str], allowed_patterns: tuple[tuple[str, ...], ...]) -> bool:
    normalized = [_normalize_token(token) for token in argv]
    for pattern in allowed_patterns:
        if len(normalized) < len(pattern):
            continue
        if all(normalized[idx] == pattern[idx] for idx in range(len(pattern))):
            return True
    return False


def _contains_dangerous_tokens(argv: list[str]) -> bool:
    normalized = [_normalize_token(token) for token in argv]
    if any(token in DANGEROUS_COMMAND_TOKENS for token in normalized):
        return True
    for idx in range(len(normalized) - 1):
        if normalized[idx] == "pip" and normalized[idx + 1] == "install":
            return True
    return False


def _safe_rel(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root)).replace("\\", "/")


def _first_non_empty(args: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(args.get(key, "")).strip()
        if value:
            return value
    return ""


def register_workspace_tools(
    registry: ToolRegistry,
    *,
    project_root: str | Path | None = None,
    shell_enabled: bool = False,
    shell_whitelist: list[tuple[str, ...]] | tuple[tuple[str, ...], ...] | None = None,
    shell_denylist: set[str] | list[str] | tuple[str, ...] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> None:
    """Register safe workspace tools."""

    allowed_patterns = tuple(shell_whitelist or DEFAULT_ALLOWED_COMMAND_PATTERNS)
    deny_tokens = set(DANGEROUS_COMMAND_TOKENS)
    if shell_denylist:
        deny_tokens.update({str(token).strip().lower() for token in shell_denylist if str(token).strip()})
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be > 0")

    def _file_list(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        root = _resolve_project_root(project_root, run_context)
        target = _resolve_path(root, str(args.get("path", ".")))
        recursive = bool(args.get("recursive", False))
        max_entries = int(args.get("max_entries", 200))
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_entries > 2000:
            max_entries = 2000
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(target.as_posix())

        entries: list[dict[str, Any]] = []
        max_mtime_ns = 0
        iterator = target.rglob("*") if recursive else target.iterdir()
        for item in iterator:
            try:
                resolved = item.resolve()
            except Exception:
                continue
            if not resolved.is_relative_to(root):
                continue
            if _is_sensitive(resolved):
                continue

            if resolved.is_file():
                if not _is_allowed_extension(resolved):
                    continue
                item_type = "file"
                size = int(resolved.stat().st_size)
                mtime_ns = int(resolved.stat().st_mtime_ns)
            elif resolved.is_dir():
                item_type = "dir"
                size = None
                mtime_ns = int(resolved.stat().st_mtime_ns)
            else:
                continue

            if mtime_ns > max_mtime_ns:
                max_mtime_ns = mtime_ns

            entries.append(
                {
                    "path": _safe_rel(resolved, root),
                    "type": item_type,
                    "size": size,
                    "mtime_ns": mtime_ns,
                }
            )
            if len(entries) >= max_entries:
                break

        snapshot_material = {
            "base_path": _safe_rel(target, root) if target != root else ".",
            "recursive": recursive,
            "entries": [
                {"path": e.get("path"), "type": e.get("type"), "size": e.get("size"), "mtime_ns": e.get("mtime_ns")}
                for e in entries
            ],
        }
        snapshot_id = _sha1_text(json.dumps(snapshot_material, ensure_ascii=False, sort_keys=True, default=str))
        snapshot_version = max_mtime_ns or int(target.stat().st_mtime_ns)

        return {
            "base_path": _safe_rel(target, root) if target != root else ".",
            "recursive": recursive,
            "count": len(entries),
            "entries": entries,
            "snapshot_id": snapshot_id,
            "snapshot_version": snapshot_version,
        }

    def _file_read(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        root = _resolve_project_root(project_root, run_context)
        path = _resolve_path(root, str(args.get("path", "")))
        size_limit = int(args.get("max_file_bytes", max_file_bytes))
        if size_limit <= 0:
            raise ValueError("max_file_bytes must be > 0")
        if _is_sensitive(path):
            raise PermissionError(f"sensitive file denied: {path.name}")
        if not _is_allowed_extension(path):
            raise PermissionError(f"file extension not allowed: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(path.as_posix())
        _ensure_file_size(path, size_limit)
        content = path.read_text(encoding="utf-8")
        mtime_ns = int(path.stat().st_mtime_ns)
        size_bytes = int(path.stat().st_size)
        content_hash = _sha1_text(content)
        snapshot_id = _sha1_text(
            json.dumps(
                {
                    "path": _safe_rel(path, root),
                    "mtime_ns": mtime_ns,
                    "size_bytes": size_bytes,
                    "content_hash": content_hash,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
        return {
            "path": str(path),
            "content": _truncate_output(content, int(args.get("max_chars", 50000))),
            "mtime_ns": mtime_ns,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
            "snapshot_id": snapshot_id,
            "snapshot_version": mtime_ns,
        }

    def _file_write(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        root = _resolve_project_root(project_root, run_context)
        path = _resolve_path(root, str(args.get("path", "")))
        size_limit = _get_max_file_bytes({**args, "max_file_bytes": args.get("max_file_bytes", max_file_bytes)})
        if _is_sensitive(path):
            raise PermissionError(f"sensitive file denied: {path.name}")
        if not _is_allowed_extension(path):
            raise PermissionError(f"file extension not allowed: {path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(args.get("content", ""))
        _ensure_content_size(content, size_limit)
        path.write_text(content, encoding="utf-8")
        return {"path": str(path), "written_chars": len(content)}

    def _file_edit(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        root = _resolve_project_root(project_root, run_context)
        path = _resolve_path(root, str(args.get("path", "")))
        size_limit = _get_max_file_bytes({**args, "max_file_bytes": args.get("max_file_bytes", max_file_bytes)})
        if _is_sensitive(path):
            raise PermissionError(f"sensitive file denied: {path.name}")
        if not _is_allowed_extension(path):
            raise PermissionError(f"file extension not allowed: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(path.as_posix())
        _ensure_file_size(path, size_limit)
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        content = path.read_text(encoding="utf-8")
        if old_text not in content:
            raise ValueError("old_text not found")
        updated = content.replace(old_text, new_text, 1)
        _ensure_content_size(updated, size_limit)
        path.write_text(updated, encoding="utf-8")
        return {"path": str(path), "edited": True}

    def _file_rename(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        root = _resolve_project_root(project_root, run_context)
        from_raw = _first_non_empty(
            args,
            "from_path",
            "source_path",
            "source",
            "old_path",
            "path_from",
        )
        to_raw = _first_non_empty(
            args,
            "to_path",
            "target_path",
            "destination_path",
            "destination",
            "new_path",
            "path_to",
        )
        from_path = _resolve_path(root, from_raw)
        to_path = _resolve_path(root, to_raw)
        if _is_sensitive(from_path) or _is_sensitive(to_path):
            raise PermissionError("sensitive path denied")
        if not _is_allowed_extension(from_path) or not _is_allowed_extension(to_path):
            raise PermissionError("file extension not allowed for rename")
        if not from_path.exists() or not from_path.is_file():
            raise FileNotFoundError(from_path.as_posix())
        if to_path.exists():
            raise FileExistsError(to_path.as_posix())
        _ensure_file_size(from_path, int(args.get("max_file_bytes", max_file_bytes)))
        to_path.parent.mkdir(parents=True, exist_ok=True)
        from_path.rename(to_path)
        return {"from_path": str(from_path), "to_path": str(to_path), "renamed": True}

    def _file_delete(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        root = _resolve_project_root(project_root, run_context)
        path_raw = _first_non_empty(args, "path", "target_path", "file_path", "target", "source_path", "source")
        path = _resolve_path(root, path_raw)
        if _is_sensitive(path):
            raise PermissionError(f"sensitive file denied: {path.name}")
        if not _is_allowed_extension(path):
            raise PermissionError(f"file extension not allowed: {path.suffix}")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path.as_posix())
        _ensure_file_size(path, int(args.get("max_file_bytes", max_file_bytes)))
        path.unlink()
        return {"path": str(path), "deleted": True}

    def _shell_exec(args: dict[str, Any], run_context: Any | None) -> dict[str, Any]:
        root = _resolve_project_root(project_root, run_context)
        effective_shell_enabled = shell_enabled
        if run_context is not None:
            profile = getattr(run_context, "permission_profile", None)
            if isinstance(profile, dict):
                effective_shell_enabled = bool(profile.get("shell_enabled", effective_shell_enabled))
        if not effective_shell_enabled:
            raise PermissionError("shell.exec disabled by default")

        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("shell.exec requires command")
        if _contains_shell_control(command):
            raise PermissionError("shell control sequences are not allowed")

        argv = _split_command(command)
        normalized_tokens = [_normalize_token(token) for token in argv]
        if any(token in deny_tokens for token in normalized_tokens):
            raise PermissionError("dangerous command token detected")
        if _contains_dangerous_tokens(argv):
            raise PermissionError("dangerous command token detected")
        if not _is_allowed_command(argv, allowed_patterns):
            raise PermissionError("command not in whitelist")

        cwd_raw = str(args.get("cwd", "."))
        cwd_path = _resolve_path(root, cwd_raw)
        if not cwd_path.exists() or not cwd_path.is_dir():
            raise PermissionError("cwd must be an existing directory under project_root")

        timeout_s = float(args.get("timeout_s", 20))
        if timeout_s <= 0 or timeout_s > 120:
            raise ValueError("timeout_s must be between 0 and 120")

        proc = subprocess.run(
            argv,
            shell=False,
            cwd=cwd_path,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return {
            "code": proc.returncode,
            "output": _truncate_output(output, int(args.get("max_output_chars", 4000))),
        }

    registry.register(
        ToolSpec(
            name="workspace.file.list",
            description="List safe files/directories under project_root.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "max_entries": {"type": "integer"},
                },
                "required": [],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "base_path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "entries": {"type": "array"},
                },
            },
            permission="workspace_read",
            timeout_s=5,
            tags=["workspace", "file", "list"],
            handler=_file_list,
        )
    )
    registry.register(
        ToolSpec(
            name="workspace.file.read",
            description="Read text file under project_root with strict safety checks.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                    "max_file_bytes": {"type": "integer"},
                },
                "required": ["path"],
            },
            output_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
            permission="workspace_read",
            timeout_s=5,
            tags=["workspace", "file", "read"],
            handler=_file_read,
        )
    )
    registry.register(
        ToolSpec(
            name="workspace.file.write",
            description="Write text file under project_root with strict safety checks.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "max_file_bytes": {"type": "integer"},
                },
                "required": ["path", "content"],
            },
            output_schema={"type": "object", "properties": {"written_chars": {"type": "integer"}}},
            permission="workspace_write",
            timeout_s=5,
            tags=["workspace", "file", "write"],
            handler=_file_write,
        )
    )
    registry.register(
        ToolSpec(
            name="workspace.file.create",
            description="Create text file under project_root (alias of workspace.file.write).",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "max_file_bytes": {"type": "integer"},
                },
                "required": ["path", "content"],
            },
            output_schema={"type": "object", "properties": {"written_chars": {"type": "integer"}}},
            permission="workspace_write",
            timeout_s=5,
            tags=["workspace", "file", "create", "write"],
            handler=_file_write,
        )
    )
    registry.register(
        ToolSpec(
            name="workspace.file.edit",
            description="Edit existing file by replacing exact old_text with new_text.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "max_file_bytes": {"type": "integer"},
                },
                "required": ["path", "old_text", "new_text"],
            },
            output_schema={"type": "object", "properties": {"edited": {"type": "boolean"}}},
            permission="workspace_write",
            timeout_s=5,
            tags=["workspace", "file", "edit"],
            handler=_file_edit,
        )
    )
    registry.register(
        ToolSpec(
            name="workspace.file.rename",
            description="Rename/move a file under project_root with strict safety checks.",
            input_schema={
                "type": "object",
                "properties": {
                    "from_path": {"type": "string"},
                    "to_path": {"type": "string"},
                    "max_file_bytes": {"type": "integer"},
                },
                "required": ["from_path", "to_path"],
            },
            output_schema={"type": "object", "properties": {"renamed": {"type": "boolean"}}},
            permission="workspace_write",
            timeout_s=5,
            tags=["workspace", "file", "rename"],
            handler=_file_rename,
        )
    )
    registry.register(
        ToolSpec(
            name="workspace.file.delete",
            description="Delete a file under project_root with strict safety checks.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_file_bytes": {"type": "integer"},
                },
                "required": ["path"],
            },
            output_schema={"type": "object", "properties": {"deleted": {"type": "boolean"}}},
            permission="workspace_write",
            timeout_s=5,
            tags=["workspace", "file", "delete"],
            handler=_file_delete,
        )
    )
    registry.register(
        ToolSpec(
            name="workspace.shell.exec",
            description="Execute whitelisted shell command under project_root.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_s": {"type": "number"},
                    "max_output_chars": {"type": "integer"},
                },
                "required": ["command"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "integer"},
                    "output": {"type": "string"},
                },
            },
            permission="shell_exec",
            timeout_s=30,
            tags=["workspace", "shell"],
            handler=_shell_exec,
        )
    )
