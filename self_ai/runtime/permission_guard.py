# coding=utf-8
"""Permission checks for ToolRuntime."""

from __future__ import annotations

from typing import Any

PERMISSIONS: tuple[str, ...] = (
    "read_only",
    "runtime_write",
    "memory_read",
    "memory_write",
    "workspace_read",
    "workspace_write",
    "shell_exec",
    "network",
    "model_call",
)


class PermissionGuard:
    """Simple policy guard for tool permissions."""

    def __init__(
        self,
        permission_profile: dict[str, bool] | None = None,
        *,
        shell_enabled: bool = False,
    ) -> None:
        # Secure-by-default profile: high-risk permissions remain disabled
        # unless explicitly enabled by caller or per-run context profile.
        defaults = {
            "read_only": True,
            "runtime_write": True,
            "memory_read": True,
            "memory_write": False,
            "workspace_read": True,
            "workspace_write": False,
            "shell_exec": shell_enabled,
            "network": False,
            "model_call": True,
        }
        self.permission_profile: dict[str, bool] = {**defaults, **(permission_profile or {})}

    def can(self, permission: str, run_context: Any | None = None) -> bool:
        if permission not in PERMISSIONS:
            return False
        profile = dict(self.permission_profile)
        if run_context is not None:
            ctx_profile = getattr(run_context, "permission_profile", None)
            if isinstance(ctx_profile, dict):
                profile.update({k: bool(v) for k, v in ctx_profile.items()})
        return bool(profile.get(permission, False))

    def deny_reason(self, permission: str) -> str:
        if permission not in PERMISSIONS:
            return f"unknown permission: {permission}"
        return f"permission denied: {permission}"
