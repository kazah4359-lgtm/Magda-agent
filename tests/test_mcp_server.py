import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.integration.mcp_server import MCPServer
from magda_agent.integration.mcp_exporter import MCPExporter

@pytest.fixture
def mock_exporter() -> MCPExporter:
    exporter = MagicMock(spec=MCPExporter)
    exporter.export_tools.return_value = [{"name": "mock_tool"}]
    async_mock = AsyncMock()
    async_mock.return_value = {"jsonrpc": "2.0", "id": 1, "result": "mock_result"}
    exporter.handle_rpc_request = async_mock
    return exporter

def test_list_tools(mock_exporter: MCPExporter) -> None:
    """Tests listing tools with default prefix."""
    server = MCPServer(mock_exporter)
    tools = server.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "magda_mock_tool"

def test_list_tools_custom_prefix(mock_exporter: MCPExporter) -> None:
    """Tests listing tools with a custom prefix."""
    server = MCPServer(mock_exporter, server_id="custom")
    tools = server.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "custom_mock_tool"

@pytest.mark.asyncio
async def test_handle_request_valid_json_prefixed(mock_exporter: MCPExporter) -> None:
    """Tests handling a valid JSON-RPC payload with prefix."""
    server = MCPServer(mock_exporter)
    payload = json.dumps({"jsonrpc": "2.0", "method": "magda_mock_tool", "id": 1})
    response_str = await server.handle_request(payload)
    response = json.loads(response_str)
    assert response["jsonrpc"] == "2.0"
    assert response["result"] == "mock_result"
    # Verify prefix was stripped before calling exporter
    mock_exporter.handle_rpc_request.assert_called_once_with({"jsonrpc": "2.0", "method": "mock_tool", "id": 1})

@pytest.mark.asyncio
async def test_handle_request_valid_json_no_prefix(mock_exporter: MCPExporter) -> None:
    """Tests handling a valid JSON-RPC payload without prefix."""
    server = MCPServer(mock_exporter)
    payload = json.dumps({"jsonrpc": "2.0", "method": "mock_tool", "id": 1})
    response_str = await server.handle_request(payload)
    response = json.loads(response_str)
    assert response["jsonrpc"] == "2.0"
    assert response["result"] == "mock_result"
    mock_exporter.handle_rpc_request.assert_called_once_with({"jsonrpc": "2.0", "method": "mock_tool", "id": 1})

@pytest.mark.asyncio
async def test_handle_request_invalid_json(mock_exporter: MCPExporter) -> None:
    """Tests handling an invalid JSON payload."""
    server = MCPServer(mock_exporter)
    payload = "invalid json"
    response_str = await server.handle_request(payload)
    response = json.loads(response_str)
    assert response["error"]["code"] == -32700
    assert response["error"]["message"] == "Parse error"
    mock_exporter.handle_rpc_request.assert_not_called()

@pytest.mark.asyncio
async def test_handle_request_notification(mock_exporter: MCPExporter) -> None:
    """Tests handling of JSON-RPC notification (no 'id' field)."""
    server = MCPServer(mock_exporter)
    payload = json.dumps({"jsonrpc": "2.0", "method": "mock_tool"})
    response_str = await server.handle_request(payload)
    # Notifications should result in no response content
    assert response_str == ""
    mock_exporter.handle_rpc_request.assert_called_once_with({"jsonrpc": "2.0", "method": "mock_tool"})

@pytest.mark.asyncio
async def test_handle_request_batch_requests(mock_exporter: MCPExporter) -> None:
    """Tests handling a batch of multiple JSON-RPC requests."""
    server = MCPServer(mock_exporter)
    payload = json.dumps([
        {"jsonrpc": "2.0", "method": "mock_tool", "id": 1},
        {"jsonrpc": "2.0", "method": "mock_tool", "id": 2}
    ])
    response_str = await server.handle_request(payload)
    responses = json.loads(response_str)
    assert isinstance(responses, list)
    assert len(responses) == 2
    assert responses[0]["result"] == "mock_result"
    assert responses[1]["result"] == "mock_result"

@pytest.mark.asyncio
async def test_handle_request_empty_batch(mock_exporter: MCPExporter) -> None:
    """Tests handling an empty batch list."""
    server = MCPServer(mock_exporter)
    payload = "[]"
    response_str = await server.handle_request(payload)
    response = json.loads(response_str)
    assert response["error"]["code"] == -32600
    assert "Invalid Request" in response["error"]["message"]
