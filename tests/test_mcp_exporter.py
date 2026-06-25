import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.integration.mcp_exporter import MCPExporter
from magda_agent.skills.registry import SkillRegistry

@pytest.fixture
def registry() -> SkillRegistry:
    """Creates a SkillRegistry fixture with dummy skills."""
    reg = SkillRegistry()

    async def dummy_skill(param1: str) -> str:
        """Dummy async skill."""
        return f"Result: {param1}"

    def dummy_sync_skill(param2: int) -> int:
        """Dummy sync skill."""
        return param2 * 2

    reg.register_skill("dummy_skill", dummy_skill, "A dummy async skill")
    reg.register_skill("dummy_sync_skill", dummy_sync_skill, "A dummy sync skill")
    return reg

@pytest.fixture
def exporter(registry: SkillRegistry) -> MCPExporter:
    """Creates an MCPExporter fixture."""
    return MCPExporter(registry)

def test_export_tools(exporter: MCPExporter) -> None:
    """Tests if tools are correctly exported via the adapter."""
    tools = exporter.export_tools()
    assert len(tools) == 2
    names = [t["name"] for t in tools]
    assert "dummy_skill" in names
    assert "dummy_sync_skill" in names

@pytest.mark.asyncio
async def test_handle_rpc_request_success(exporter: MCPExporter) -> None:
    """Tests a successful JSON-RPC tool execution request."""
    req = {
        "jsonrpc": "2.0",
        "method": "dummy_skill",
        "params": {"param1": "test"},
        "id": 1
    }
    res = await exporter.handle_rpc_request(req)
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert "result" in res
    assert not res["result"]["isError"]
    assert res["result"]["content"][0]["text"] == "Result: test"

@pytest.mark.asyncio
async def test_handle_rpc_request_invalid_method(exporter: MCPExporter) -> None:
    """Tests handling of an invalid JSON-RPC method."""
    req = {
        "jsonrpc": "2.0",
        "method": "non_existent_skill",
        "params": {},
        "id": 2
    }
    res = await exporter.handle_rpc_request(req)
    assert "error" in res
    assert res["error"]["code"] == -32601

@pytest.mark.asyncio
async def test_handle_rpc_request_invalid_jsonrpc(exporter: MCPExporter) -> None:
    """Tests handling of an invalid JSON-RPC version."""
    req = {
        "method": "dummy_skill",
        "params": {"param1": "test"},
        "id": 3
    }
    res = await exporter.handle_rpc_request(req)
    assert "error" in res
    assert res["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_handle_rpc_request_error_in_tool(exporter: MCPExporter) -> None:
    """Tests error handling when the underlying tool raises an exception."""
    req = {
        "jsonrpc": "2.0",
        "method": "dummy_skill",
        "params": {}, # Missing required param
        "id": 4
    }
    res = await exporter.handle_rpc_request(req)
    # The adapter wraps it, but the exporter now properly returns a JSON-RPC error.
    assert "error" in res
    assert res["error"]["code"] == -32000
    assert res["error"]["message"].startswith("Error executing skill")
