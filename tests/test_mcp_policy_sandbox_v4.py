"""Tests for MCP Tool Runtime Execution Policy Sandbox V4."""

import pytest

from magda_agent.safety.mcp_policy_sandbox_v4 import (
    MCPPolicySandboxV4,
    MCPPolicySandboxViolationError,
    SandboxPolicyRule,
    SideEffectPolicyRule,
    BlockedToolsPolicyRule,
)


def sample_read_tool(file_path: str) -> str:
    """Sample read-only tool function."""
    return f"Content of {file_path}"


def sample_write_tool(file_path: str, content: str, is_verified: bool = False) -> str:
    """Sample state-mutating write tool function."""
    return f"Wrote {len(content)} bytes to {file_path}"


async def sample_async_delete_tool(user_id: int, is_verified: bool = False) -> str:
    """Sample async state-mutating delete tool function."""
    return f"Deleted user {user_id}"


class MockAuditTrail:
    """Mock audit trail logger."""

    def __init__(self) -> None:
        self.logs = []

    def log_event(self, event_type: str, data: dict) -> None:
        self.logs.append((event_type, data))


class CustomParamCheckRule(SandboxPolicyRule):
    """Custom rule blocking executions with unsafe parameter values."""

    def evaluate(self, tool_name: str, **kwargs) -> tuple[bool, str]:
        if "forbidden" in str(kwargs.get("content", "")).lower():
            return False, "Content contains forbidden keywords."
        return True, "Content check passed."


def test_read_tool_allowed_by_default() -> None:
    """Verify read-only tools run without requiring verification."""
    sandbox = MCPPolicySandboxV4()
    result = sandbox.execute(sample_read_tool, "get_info", file_path="config.json")
    assert result == "Content of config.json"


def test_unverified_side_effect_blocked() -> None:
    """Verify action tools with side-effects are blocked when unverified."""
    sandbox = MCPPolicySandboxV4()
    with pytest.raises(MCPPolicySandboxViolationError) as exc_info:
        sandbox.execute(sample_write_tool, "write_file", file_path="test.txt", content="hello")
    assert "Unverified side-effect blocked" in str(exc_info.value)


def test_verified_side_effect_allowed() -> None:
    """Verify state-mutating action tools pass when verified."""
    sandbox = MCPPolicySandboxV4()
    result = sandbox.execute(
        sample_write_tool,
        "write_file",
        file_path="test.txt",
        content="hello",
        is_verified=True,
    )
    assert result == "Wrote 5 bytes to test.txt"


def test_explicitly_blocked_tools() -> None:
    """Verify explicitly blocked tools are denied execution."""
    sandbox = MCPPolicySandboxV4(blocked_tools=["get_info"])
    with pytest.raises(MCPPolicySandboxViolationError) as exc_info:
        sandbox.execute(sample_read_tool, "get_info", file_path="data.csv")
    assert "explicitly blocked by policy" in str(exc_info.value)


def test_custom_policy_rule() -> None:
    """Verify custom sandbox policy rules are enforced."""
    sandbox = MCPPolicySandboxV4()
    sandbox.add_rule(CustomParamCheckRule())

    # Allowed when content is safe and verified
    res = sandbox.execute(
        sample_write_tool,
        "write_file",
        file_path="a.txt",
        content="safe content",
        is_verified=True,
    )
    assert "Wrote 12 bytes" in res

    # Blocked by custom rule when forbidden content present
    with pytest.raises(MCPPolicySandboxViolationError) as exc_info:
        sandbox.execute(
            sample_write_tool,
            "write_file",
            file_path="a.txt",
            content="contains forbidden word",
            is_verified=True,
        )
    assert "forbidden keywords" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_execution() -> None:
    """Verify async tool execution and side-effect policy checks."""
    sandbox = MCPPolicySandboxV4()

    # Blocked when unverified
    with pytest.raises(MCPPolicySandboxViolationError):
        await sandbox.execute_async(sample_async_delete_tool, "delete_user", user_id=42)

    # Allowed when verified
    res = await sandbox.execute_async(
        sample_async_delete_tool, "delete_user", user_id=42, is_verified=True
    )
    assert res == "Deleted user 42"


def test_wrapper_decorator_sync() -> None:
    """Verify sync tool decorator function wrapping."""
    sandbox = MCPPolicySandboxV4()

    @sandbox.wrap_tool()
    def write_data(file_path: str, content: str, is_verified: bool = False) -> str:
        return f"Saved to {file_path}"

    with pytest.raises(MCPPolicySandboxViolationError):
        write_data(file_path="db.sqlite", content="data")

    result = write_data(file_path="db.sqlite", content="data", is_verified=True)
    assert result == "Saved to db.sqlite"


@pytest.mark.asyncio
async def test_wrapper_decorator_async() -> None:
    """Verify async tool decorator function wrapping."""
    sandbox = MCPPolicySandboxV4()

    @sandbox.wrap_tool("delete_item")
    async def remove_item(item_id: int, is_verified: bool = False) -> str:
        return f"Removed {item_id}"

    with pytest.raises(MCPPolicySandboxViolationError):
        await remove_item(item_id=101)

    result = await remove_item(item_id=101, is_verified=True)
    assert result == "Removed 101"


def test_audit_trail_logging() -> None:
    """Verify sandbox logs execution attempts to audit trail."""
    audit = MockAuditTrail()
    sandbox = MCPPolicySandboxV4(audit_trail=audit)

    # Blocked call
    with pytest.raises(MCPPolicySandboxViolationError):
        sandbox.execute(sample_write_tool, "write_config", file_path="a.cfg", content="x")

    # Allowed call
    sandbox.execute(sample_read_tool, "read_config", file_path="a.cfg")

    assert len(audit.logs) == 2
    event_type1, log1 = audit.logs[0]
    assert event_type1 == "mcp_policy_sandbox"
    assert log1["status"] == "blocked"
    assert log1["tool_name"] == "write_config"

    event_type2, log2 = audit.logs[1]
    assert event_type2 == "mcp_policy_sandbox"
    assert log2["status"] == "allowed"
    assert log2["tool_name"] == "read_config"


def test_type_mismatch_raises_value_error() -> None:
    """Verify executing coroutine with execute() or sync function with execute_async() raises ValueError."""
    sandbox = MCPPolicySandboxV4()

    with pytest.raises(ValueError) as exc1:
        sandbox.execute(sample_async_delete_tool, "delete_user", user_id=1)
    assert "does not support coroutines" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:

        @pytest.mark.asyncio
        async def dummy_runner():
            await sandbox.execute_async(sample_read_tool, "read_tool", file_path="x")

        import asyncio

        asyncio.run(dummy_runner())
    assert "requires a coroutine function" in str(exc2.value)
