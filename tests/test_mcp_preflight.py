import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.integration.mcp_preflight import MCPPreflightValidator, PreflightMCPServerWrapper
from magda_agent.integration.mcp_server import MCPServer
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.taint import mark_tainted


@pytest.fixture
def mock_registry() -> SkillRegistry:
    """Fixture for a mocked SkillRegistry."""
    registry = MagicMock(spec=SkillRegistry)
    registry.has_skill.side_effect = lambda name: name in ["get_weather", "add_numbers"]
    return registry


@pytest.fixture
def mock_server() -> MCPServer:
    """Fixture for a mocked MCPServer."""
    server = MagicMock(spec=MCPServer)
    server.handle_request = AsyncMock(return_value=json.dumps({
        "jsonrpc": "2.0",
        "id": "1",
        "result": "success"
    }))
    # Mock server.exporter.registry
    server.exporter = MagicMock()
    server.exporter.registry = MagicMock(spec=SkillRegistry)
    server.exporter.registry.has_skill.side_effect = lambda name: name in ["get_weather"]
    return server


def test_validator_valid_request(mock_registry: SkillRegistry) -> None:
    """Tests that a standard, safe request dictionary passes validation."""
    validator = MCPPreflightValidator(registry=mock_registry)
    request = {
        "jsonrpc": "2.0",
        "method": "get_weather",
        "params": {"location": "San Francisco"},
        "id": "1"
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is True
    assert err_code == 0
    assert err_msg == ""


def test_validator_invalid_jsonrpc() -> None:
    """Tests that an invalid JSON-RPC version is rejected."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "1.0",
        "method": "get_weather",
        "params": {}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32600
    assert "jsonrpc version must be '2.0'" in err_msg


def test_validator_invalid_method() -> None:
    """Tests that a missing or invalid method is rejected."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "2.0",
        "params": {}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32600
    assert "method" in err_msg


def test_validator_tainted_method() -> None:
    """Tests that a request with a tainted tool name (method) is blocked."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "2.0",
        "method": mark_tainted("unsafe_tool"),
        "params": {}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32000
    assert "tool name is tainted" in err_msg


def test_validator_forbidden_tool() -> None:
    """Tests that blacklisted tools are blocked."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "2.0",
        "method": "forbidden_tool",
        "params": {}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32000
    assert "blacklisted" in err_msg


def test_validator_tainted_params() -> None:
    """Tests that a request with tainted arguments in params is blocked."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "2.0",
        "method": "get_weather",
        "params": {"location": mark_tainted("New York")}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32000
    assert "Tainted data detected" in err_msg


def test_validator_hazardous_shell_injection() -> None:
    """Tests that hazardous shell injection commands are blocked."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "2.0",
        "method": "get_weather",
        "params": {"location": "San Francisco; rm -rf /"}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32000
    assert "Hazardous shell pattern detected" in err_msg


def test_validator_path_traversal() -> None:
    """Tests that path traversal attempts are blocked."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "2.0",
        "method": "get_weather",
        "params": {"location": "../../../etc/passwd"}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32000
    assert "Hazardous path traversal pattern" in err_msg


def test_validator_sql_injection() -> None:
    """Tests that SQL injection strings are blocked."""
    validator = MCPPreflightValidator()
    request = {
        "jsonrpc": "2.0",
        "method": "get_weather",
        "params": {"location": "' OR 1=1 --"}
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32000
    assert "Hazardous SQL injection pattern" in err_msg


def test_validator_unregistered_method(mock_registry: SkillRegistry) -> None:
    """Tests that unregistered tools are rejected when SkillRegistry is active."""
    validator = MCPPreflightValidator(registry=mock_registry)
    request = {
        "jsonrpc": "2.0",
        "method": "non_existent_tool",
        "params": {},
        "id": "1"
    }
    is_valid, err_code, err_msg = validator.validate_request_dict(request)
    assert is_valid is False
    assert err_code == -32601
    assert "not registered" in err_msg


@pytest.mark.asyncio
async def test_wrapper_passes_safe_payload(mock_server: MCPServer) -> None:
    """Tests that the wrapper forwards safe payloads to the underlying server."""
    wrapper = PreflightMCPServerWrapper(server=mock_server)
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "get_weather",
        "params": {"location": "Seattle"},
        "id": "abc"
    })
    response_str = await wrapper.handle_request(payload)
    response = json.loads(response_str)
    assert response["result"] == "success"
    mock_server.handle_request.assert_called_once_with(payload)


@pytest.mark.asyncio
async def test_wrapper_blocks_unsafe_payload(mock_server: MCPServer) -> None:
    """Tests that the wrapper intercepts unsafe payloads and returns JSON-RPC error response."""
    wrapper = PreflightMCPServerWrapper(server=mock_server)
    payload = json.dumps({
        "jsonrpc": "2.0",
        "method": "forbidden_tool",
        "params": {},
        "id": "xyz"
    })
    response_str = await wrapper.handle_request(payload)
    response = json.loads(response_str)
    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "blacklisted" in response["error"]["message"]
    mock_server.handle_request.assert_not_called()


@pytest.mark.asyncio
async def test_wrapper_invalid_json_parsing(mock_server: MCPServer) -> None:
    """Tests that invalid raw JSON payloads are intercepted at the parser layer."""
    wrapper = PreflightMCPServerWrapper(server=mock_server)
    payload = "{invalid_json}"
    response_str = await wrapper.handle_request(payload)
    response = json.loads(response_str)
    assert "error" in response
    assert response["error"]["code"] == -32700
    assert "Parse error" in response["error"]["message"]
    mock_server.handle_request.assert_not_called()
