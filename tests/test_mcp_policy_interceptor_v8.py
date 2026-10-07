"""
Unit tests for MCP Action Tool Policy Interceptor V8.
"""

import pytest
import asyncio
from typing import Any, Dict, List

from magda_agent.skills.mcp_policy_interceptor_v8 import (
    MCPActionToolPolicyInterceptorV8,
    MCPPolicyInterceptorV8,
    MCPPolicyViolationErrorV8,
)
from magda_agent.safety.taint import mark_tainted, TaintedString


class MockAuditTrail:
    """Mock audit trail for capturing policy events."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append({"event_type": event_type, "data": data})

    def append(self, entry: Dict[str, Any]) -> None:
        self.events.append({"event_type": "append", "data": entry})


def test_valid_sync_tool_execution() -> None:
    """Verify valid synchronous tool execution passes policy checks."""
    interceptor = MCPActionToolPolicyInterceptorV8()

    def sample_tool(command: str, count: int = 1) -> str:
        return f"Executed {command} {count} times"

    result = interceptor.execute(sample_tool, "sample_tool", command="echo 'hello'", count=2)
    assert result == "Executed echo 'hello' 2 times"


@pytest.mark.asyncio
async def test_valid_async_tool_execution() -> None:
    """Verify valid asynchronous tool execution passes policy checks."""
    interceptor = MCPActionToolPolicyInterceptorV8()

    async def sample_async_tool(payload: str) -> str:
        await asyncio.sleep(0.001)
        return f"Processed {payload}"

    result = await interceptor.execute_async(sample_async_tool, "sample_async_tool", payload="normal_data")
    assert result == "Processed normal_data"


def test_explicitly_blocked_tool() -> None:
    """Verify calling an explicitly blocked tool raises MCPPolicyViolationErrorV8."""
    interceptor = MCPActionToolPolicyInterceptorV8(blocked_tools=["danger_tool"])

    def danger_tool() -> str:
        return "should not run"

    with pytest.raises(MCPPolicyViolationErrorV8) as exc_info:
        interceptor.execute(danger_tool, "danger_tool")

    assert "explicitly forbidden" in str(exc_info.value)


def test_forbidden_external_host_in_payload() -> None:
    """Verify payloads containing URLs targeting blocked hosts raise MCPPolicyViolationErrorV8."""
    interceptor = MCPActionToolPolicyInterceptorV8(blocked_hosts=["malicious.com"])

    def http_fetch(url: str) -> str:
        return "fetched"

    with pytest.raises(MCPPolicyViolationErrorV8) as exc_info:
        interceptor.execute(http_fetch, "http_fetch", url="https://malicious.com/api/steal")

    assert "forbidden host" in str(exc_info.value)


def test_dangerous_pattern_in_payload() -> None:
    """Verify payloads containing dangerous commands raise MCPPolicyViolationErrorV8."""
    interceptor = MCPActionToolPolicyInterceptorV8()

    def run_command(script: str) -> str:
        return "ran"

    with pytest.raises(MCPPolicyViolationErrorV8) as exc_info:
        interceptor.execute(run_command, "run_command", script="rm -rf /tmp/data")

    assert "Dangerous pattern" in str(exc_info.value)


def test_tainted_payload_blocked() -> None:
    """Verify tainted inputs marked with TaintedString or __tainted__ flag are blocked."""
    interceptor = MCPActionToolPolicyInterceptorV8()

    tainted_param = mark_tainted("user_input_from_untrusted_source")

    def process_input(data: Any) -> str:
        return "processed"

    with pytest.raises(MCPPolicyViolationErrorV8) as exc_info:
        interceptor.execute(process_input, "process_input", data=tainted_param)

    assert "Tainted" in str(exc_info.value) or "blocked" in str(exc_info.value)

    # Test explicit dict taint flag
    with pytest.raises(MCPPolicyViolationErrorV8):
        interceptor.execute(process_input, "process_input", data={"config": "val", "__tainted__": True})


def test_wrap_tool_decorator_sync_and_async() -> None:
    """Verify wrap_tool decorator correctly intercepts sync and async functions."""
    audit = MockAuditTrail()
    interceptor = MCPActionToolPolicyInterceptorV8(audit_trail=audit)

    @interceptor.wrap_tool(tool_name="safe_sync")
    def safe_sync_fn(text: str) -> str:
        return text.upper()

    @interceptor.wrap_tool(tool_name="safe_async")
    async def safe_async_fn(text: str) -> str:
        return text.lower()

    assert safe_sync_fn(text="hello") == "HELLO"
    assert asyncio.run(safe_async_fn(text="WORLD")) == "world"

    # Ensure audit events logged
    assert len(audit.events) == 2
    assert audit.events[0]["data"]["tool_name"] == "safe_sync"
    assert audit.events[0]["data"]["status"] == "allowed"


def test_dynamic_configuration_and_aliases() -> None:
    """Verify dynamic adding of blocked tools/hosts/patterns and alias compatibility."""
    interceptor = MCPPolicyInterceptorV8()

    interceptor.add_blocked_tool("new_blocked_tool")
    interceptor.add_blocked_host("evil.com")
    interceptor.add_blocked_pattern("drop database")

    def dummy() -> None:
        pass

    with pytest.raises(MCPPolicyViolationErrorV8):
        interceptor.execute(dummy, "new_blocked_tool")

    with pytest.raises(MCPPolicyViolationErrorV8):
        interceptor.execute(dummy, "some_tool", url="http://evil.com/x")

    with pytest.raises(MCPPolicyViolationErrorV8):
        interceptor.execute(dummy, "some_tool", query="DROP DATABASE users;")
