"""
Unit tests for MCPToolRegistryAuthInterceptorV2 in magda_agent/safety/mcp_interceptor_v2.py.
"""

import pytest
from unittest.mock import MagicMock

from magda_agent.safety.mcp_interceptor_v2 import (
    MCPToolRegistryAuthInterceptorV2,
    MCPAuthInterceptorV2,
    MCPAuthInterceptorError,
)
from magda_agent.skills.mcp_registry import MCPRegistry, MCPRegistrationError


@pytest.fixture
def interceptor() -> MCPToolRegistryAuthInterceptorV2:
    return MCPToolRegistryAuthInterceptorV2(
        required_tokens={"write_file": "secret-token-123"},
        required_roles={"admin_tool": "admin"},
        blocked_tools=["forbidden_tool"],
        blocked_hosts=["malicious.com", "phishing.net"],
    )


def test_registry_integration_valid_schema(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    registry = MCPRegistry()
    registry.add_interceptor(interceptor.validate_schema_interceptor)

    valid_schema = {
        "name": "safe_calculator",
        "description": "Performs addition and subtraction on safe numbers.",
        "inputSchema": {"type": "object"},
    }

    assert registry.load_tool(valid_schema) is True
    assert "safe_calculator" in registry.list_tools()


def test_registry_integration_blocked_tool(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    registry = MCPRegistry()
    registry.add_interceptor(interceptor.validate_schema_interceptor)

    blocked_schema = {
        "name": "forbidden_tool",
        "description": "A forbidden tool.",
        "inputSchema": {"type": "object"},
    }

    with pytest.raises(MCPRegistrationError, match="forbidden_tool"):
        registry.load_tool(blocked_schema)


def test_registry_integration_forbidden_host_in_description(
    interceptor: MCPToolRegistryAuthInterceptorV2,
) -> None:
    registry = MCPRegistry()
    registry.add_interceptor(interceptor.validate_schema_interceptor)

    malicious_schema = {
        "name": "external_fetcher",
        "description": "Fetches data from http://malicious.com/api",
        "inputSchema": {"type": "object"},
    }

    with pytest.raises(MCPRegistrationError, match="malicious.com"):
        registry.load_tool(malicious_schema)


def test_intercept_execution_allowed_safe_call(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    result = interceptor.intercept_execution("read_file", {"filepath": "doc.txt"})
    assert result is True


def test_intercept_execution_allowed_with_valid_token(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    result = interceptor.intercept_execution(
        "write_file",
        {"filepath": "out.txt", "content": "hello"},
        auth_token="secret-token-123",
    )
    assert result is True


def test_intercept_execution_allowed_with_prefix_token(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    result = interceptor.intercept_execution(
        "write_file",
        {"filepath": "out.txt"},
        auth_token="secret-token-123:session-1",
    )
    assert result is True


def test_intercept_execution_blocked_missing_token(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    with pytest.raises(MCPAuthInterceptorError, match="requires an auth token"):
        interceptor.intercept_execution("write_file", {"filepath": "out.txt"})


def test_intercept_execution_blocked_invalid_token(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    with pytest.raises(MCPAuthInterceptorError, match="Invalid auth token"):
        interceptor.intercept_execution("write_file", {"filepath": "out.txt"}, auth_token="wrong-token")


def test_intercept_execution_allowed_with_valid_role(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    result = interceptor.intercept_execution("admin_tool", {}, user_role="admin")
    assert result is True


def test_intercept_execution_blocked_insufficient_role(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    with pytest.raises(MCPAuthInterceptorError, match="insufficient"):
        interceptor.intercept_execution("admin_tool", {}, user_role="viewer")


def test_intercept_execution_blocked_missing_role(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    with pytest.raises(MCPAuthInterceptorError, match="requires user role"):
        interceptor.intercept_execution("admin_tool", {})


def test_intercept_execution_blocked_forbidden_tool(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    with pytest.raises(MCPAuthInterceptorError, match="explicitly forbidden"):
        interceptor.intercept_execution("forbidden_tool", {})


def test_intercept_execution_blocked_unsafe_external_host(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    payload = {"target_url": "https://phishing.net/login"}
    with pytest.raises(MCPAuthInterceptorError, match="Unsafe external call target 'phishing.net'"):
        interceptor.intercept_execution("http_get", payload)


def test_intercept_execution_blocked_unsafe_command_pattern(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    payload = {"command": "rm -rf /var/data"}
    with pytest.raises(MCPAuthInterceptorError, match="Unsafe command pattern 'rm -rf'"):
        interceptor.intercept_execution("execute_cmd", payload)


def test_wrap_tool_sync(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    @interceptor.wrap_tool("custom_write", required_token="custom-secret")
    def sync_tool(filepath: str) -> str:
        return f"wrote to {filepath}"

    # Execution blocked without token
    with pytest.raises(MCPAuthInterceptorError, match="requires an auth token"):
        sync_tool(filepath="a.txt")

    # Execution succeeds with token
    res = sync_tool(filepath="a.txt", auth_token="custom-secret")
    assert res == "wrote to a.txt"


@pytest.mark.asyncio
async def test_wrap_tool_async(interceptor: MCPToolRegistryAuthInterceptorV2) -> None:
    @interceptor.wrap_tool("async_write", required_token="async-secret")
    async def async_tool(filepath: str) -> str:
        return f"async wrote to {filepath}"

    # Execution blocked without token
    with pytest.raises(MCPAuthInterceptorError, match="requires an auth token"):
        await async_tool(filepath="b.txt")

    # Execution succeeds with token
    res = await async_tool(filepath="b.txt", auth_token="async-secret")
    assert res == "async wrote to b.txt"


def test_audit_trail_logging() -> None:
    mock_audit = MagicMock()
    interceptor = MCPToolRegistryAuthInterceptorV2(
        required_tokens={"protected": "tok123"},
        audit_trail=mock_audit,
    )

    interceptor.intercept_execution("protected", {}, auth_token="tok123")
    assert mock_audit.log_event.called or mock_audit.log_call.called or mock_audit.append.called
