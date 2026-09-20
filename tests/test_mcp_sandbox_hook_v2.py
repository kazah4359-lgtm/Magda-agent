"""Tests for MCP Dynamic Verification Sandbox Hook V2."""

from unittest.mock import MagicMock
import pytest

from magda_agent.safety.mcp_sandbox_hook_v2 import (
    MCPDynamicSandboxHookV2,
    MCPDynamicSandboxViolationError,
    MCPSandboxHookV2,
)


def sample_read_tool(item_id: int) -> str:
    """Read-only tool."""
    return f"Item {item_id} details"


def sample_external_api_tool(url: str, payload: dict, is_verified: bool = False) -> str:
    """External API call tool."""
    return f"Response from {url}"


async def sample_async_external_tool(endpoint: str, data: dict, is_verified: bool = False) -> str:
    """Async external tool."""
    return f"Async response from {endpoint}"


class MockAuditTrail:
    """Mock audit trail logger."""

    def __init__(self) -> None:
        self.events = []

    def log_event(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


def test_alias_and_instantiation() -> None:
    """Verify MCPSandboxHookV2 alias and basic class properties."""
    hook = MCPSandboxHookV2()
    assert isinstance(hook, MCPDynamicSandboxHookV2)
    assert hook.strict_verification is True


def test_sync_tool_execution_allowed() -> None:
    """Verify internal / non-external read tools execute without extra verification."""
    sandbox = MCPDynamicSandboxHookV2()
    res = sandbox.execute(sample_read_tool, "read_item", item_id=123)
    assert res == "Item 123 details"


def test_external_tool_blocked_when_unverified() -> None:
    """Verify external tools require verification in strict mode."""
    sandbox = MCPDynamicSandboxHookV2()

    with pytest.raises(MCPDynamicSandboxViolationError) as exc_info:
        sandbox.execute(
            sample_external_api_tool,
            "external_api_call",
            url="https://api.example.com/v1",
            payload={"key": "val"},
        )
    assert "requires explicit verification" in str(exc_info.value)


def test_external_tool_allowed_when_verified() -> None:
    """Verify external tools pass when explicitly verified."""
    sandbox = MCPDynamicSandboxHookV2()
    res = sandbox.execute(
        sample_external_api_tool,
        "external_api_call",
        url="https://api.example.com/v1",
        payload={"key": "val"},
        is_verified=True,
    )
    assert res == "Response from https://api.example.com/v1"


def test_blocked_tools_denied() -> None:
    """Verify explicitly blocked tools fail verification."""
    sandbox = MCPDynamicSandboxHookV2(blocked_tools=["read_item"])

    with pytest.raises(MCPDynamicSandboxViolationError) as exc_info:
        sandbox.execute(sample_read_tool, "read_item", item_id=1)
    assert "explicitly blocked" in str(exc_info.value)


def test_blocked_host_denied_in_payload() -> None:
    """Verify external calls referencing blocked hosts are denied."""
    sandbox = MCPDynamicSandboxHookV2(blocked_hosts=["phishing.net"])

    with pytest.raises(MCPDynamicSandboxViolationError) as exc_info:
        sandbox.execute(
            sample_external_api_tool,
            "fetch_data",
            url="https://phishing.net/login",
            payload={},
            is_verified=True,
        )
    assert "Unsafe external target 'phishing.net'" in str(exc_info.value)


def test_pre_and_post_execution_hooks_sync() -> None:
    """Verify pre-execution and post-execution hooks run and can mock / control execution."""
    pre_hook_mock = MagicMock(return_value=(True, "Hook approval"))
    post_hook_mock = MagicMock()

    sandbox = MCPDynamicSandboxHookV2(
        pre_hooks=[pre_hook_mock],
        post_hooks=[post_hook_mock],
    )

    result = sandbox.execute(sample_read_tool, "get_data", item_id=99)
    assert result == "Item 99 details"

    pre_hook_mock.assert_called_once_with("get_data", {"item_id": 99})
    post_hook_mock.assert_called_once_with("get_data", "Item 99 details", {"item_id": 99})


def test_pre_execution_hook_denial() -> None:
    """Verify pre-execution hook can deny execution."""
    denying_hook = MagicMock(return_value=(False, "Security policy failure"))

    sandbox = MCPDynamicSandboxHookV2(pre_hooks=[denying_hook])

    with pytest.raises(MCPDynamicSandboxViolationError) as exc_info:
        sandbox.execute(sample_read_tool, "get_data", item_id=1)

    assert "Security policy failure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_execution_and_async_hooks() -> None:
    """Verify async tool execution and async pre/post hooks."""
    async_pre_mock = MagicMock()
    async_pre_mock.__name__ = "async_pre_hook"

    async def async_pre_hook(tool_name: str, payload: dict) -> tuple[bool, str]:
        async_pre_mock(tool_name, payload)
        return True, "Async pre-hook passed"

    async_post_mock = MagicMock()
    async_post_mock.__name__ = "async_post_hook"

    async def async_post_hook(tool_name: str, result: str, payload: dict) -> None:
        async_post_mock(tool_name, result, payload)

    sandbox = MCPDynamicSandboxHookV2(
        pre_hooks=[async_pre_hook],
        post_hooks=[async_post_hook],
    )

    res = await sandbox.execute_async(
        sample_async_external_tool,
        "external_async_fetch",
        endpoint="/api/status",
        data={},
        is_verified=True,
    )

    assert res == "Async response from /api/status"
    async_pre_mock.assert_called_once_with(
        "external_async_fetch",
        {"endpoint": "/api/status", "data": {}, "is_verified": True},
    )
    async_post_mock.assert_called_once_with(
        "external_async_fetch",
        "Async response from /api/status",
        {"endpoint": "/api/status", "data": {}, "is_verified": True},
    )


def test_decorator_sync_and_async() -> None:
    """Verify wrap_tool decorator functionality for sync and async tools."""
    sandbox = MCPDynamicSandboxHookV2()

    @sandbox.wrap_tool()
    def my_sync_tool(item_id: int) -> str:
        return f"Sync {item_id}"

    @sandbox.wrap_tool("external_async_tool")
    async def my_async_tool(url: str, is_verified: bool = False) -> str:
        return f"Fetched {url}"

    assert my_sync_tool(item_id=5) == "Sync 5"

    with pytest.raises(MCPDynamicSandboxViolationError):
        import asyncio

        asyncio.run(my_async_tool(url="https://example.com"))

    res = asyncio.run(my_async_tool(url="https://example.com", is_verified=True))
    assert res == "Fetched https://example.com"


def test_audit_trail_logging() -> None:
    """Verify audit log event emission for allowed and blocked executions."""
    audit = MockAuditTrail()
    sandbox = MCPDynamicSandboxHookV2(audit_trail=audit)

    sandbox.execute(sample_read_tool, "read_item", item_id=1)

    with pytest.raises(MCPDynamicSandboxViolationError):
        sandbox.execute(
            sample_external_api_tool,
            "external_call",
            url="http://malicious.com/api",
            payload={},
            is_verified=True,
        )

    assert len(audit.events) == 2
    assert audit.events[0][0] == "mcp_sandbox_hook_v2"
    assert audit.events[0][1]["status"] == "allowed"

    assert audit.events[1][0] == "mcp_sandbox_hook_v2"
    assert audit.events[1][1]["status"] == "blocked"


def test_dynamic_hook_and_rule_management() -> None:
    """Verify dynamic addition and removal of hooks and block rules."""
    sandbox = MCPDynamicSandboxHookV2()

    dummy_hook = lambda name, payload: True
    sandbox.add_pre_hook(dummy_hook)
    assert dummy_hook in sandbox.pre_hooks

    sandbox.remove_pre_hook(dummy_hook)
    assert dummy_hook not in sandbox.pre_hooks

    sandbox.add_blocked_tool("forbidden_tool")
    assert "forbidden_tool" in sandbox.blocked_tools

    sandbox.add_blocked_host("bad-host.org")
    assert "bad-host.org" in sandbox.blocked_hosts


def test_mismatched_sync_async_execution_type_error() -> None:
    """Verify TypeError or ValueError when sync function passed to execute_async or coroutine to execute."""
    sandbox = MCPDynamicSandboxHookV2()

    with pytest.raises(ValueError) as exc1:
        sandbox.execute(sample_async_external_tool, "async_tool", endpoint="/test", data={})
    assert "does not support coroutines" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        import asyncio

        asyncio.run(
            sandbox.execute_async(sample_read_tool, "sync_tool", item_id=10)
        )
    assert "requires a coroutine function" in str(exc2.value)
