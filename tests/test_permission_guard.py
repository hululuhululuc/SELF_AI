# coding=utf-8
"""Tests for ToolRuntime permission guard."""

from self_ai.runtime.permission_guard import PermissionGuard


def test_permission_guard_allow_default_read() -> None:
    guard = PermissionGuard()
    assert guard.can("read_only") is True
    assert guard.can("model_call") is True


def test_permission_guard_deny_shell_by_default() -> None:
    guard = PermissionGuard()
    assert guard.can("shell_exec") is False


def test_permission_guard_deny_workspace_write_by_default() -> None:
    guard = PermissionGuard()
    assert guard.can("workspace_write") is False


def test_permission_guard_deny_network_by_default() -> None:
    guard = PermissionGuard()
    assert guard.can("network") is False


def test_permission_guard_custom_profile_override() -> None:
    guard = PermissionGuard(
        permission_profile={
            "memory_write": True,
            "network": True,
            "workspace_write": True,
            "shell_exec": True,
        }
    )
    assert guard.can("memory_write") is True
    assert guard.can("network") is True
    assert guard.can("workspace_write") is True
    assert guard.can("shell_exec") is True
