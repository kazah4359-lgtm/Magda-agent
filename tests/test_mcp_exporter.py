import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, AsyncMock
from magda_agent.skills.mcp_exporter import MCPSkillExporter
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

    def complex_skill(a: int, b: float, c: bool, d: list, e: dict, f: str = "default"):
        """A skill with complex types."""
        return "complex"

    reg.register_skill("complex_skill", complex_skill, "A skill with complex types")

    return reg

@pytest.fixture
def exporter(registry: SkillRegistry) -> MCPSkillExporter:
    """Creates an MCPSkillExporter fixture."""
    return MCPSkillExporter(registry)

def test_export_tools(exporter: MCPSkillExporter) -> None:
    """Tests if tools are correctly exported via the adapter."""
    tools: List[Dict[str, Any]] = exporter.list_tools()
    assert len(tools) == 3
    names: List[str] = [t["name"] for t in tools]
    assert "dummy_skill" in names
    assert "dummy_sync_skill" in names
    assert "complex_skill" in names

    # Verify complex_skill schema
    complex_tool = next(t for t in tools if t["name"] == "complex_skill")
    schema = complex_tool["inputSchema"]
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["a"]["type"] == "integer"
    assert props["b"]["type"] == "number"
    assert props["c"]["type"] == "boolean"
    assert props["d"]["type"] == "array"
    assert props["e"]["type"] == "object"
    assert props["f"]["type"] == "string"

    assert "a" in schema["required"]
    assert "f" not in schema["required"]

@pytest.mark.asyncio
async def test_handle_rpc_request_success(exporter: MCPSkillExporter) -> None:
    """Tests a successful JSON-RPC tool execution request."""
    req = {
        "jsonrpc": "2.0",
        "method": "dummy_skill",
        "params": {"param1": "test"},
        "id": 1
    }
    res: Dict[str, Any] = await exporter.handle_rpc_request(req)
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert "result" in res
    assert not res["result"]["isError"]
    assert res["result"]["content"][0]["text"] == "Result: test"

@pytest.mark.asyncio
async def test_handle_rpc_request_invalid_method(exporter: MCPSkillExporter) -> None:
    """Tests handling of an invalid JSON-RPC method."""
    req = {
        "jsonrpc": "2.0",
        "method": "non_existent_skill",
        "params": {},
        "id": 2
    }
    res: Dict[str, Any] = await exporter.handle_rpc_request(req)
    assert "error" in res
    assert res["error"]["code"] == -32601

@pytest.mark.asyncio
async def test_handle_rpc_request_invalid_jsonrpc(exporter: MCPSkillExporter) -> None:
    """Tests handling of an invalid JSON-RPC version."""
    req = {
        "method": "dummy_skill",
        "params": {"param1": "test"},
        "id": 3
    }
    res: Dict[str, Any] = await exporter.handle_rpc_request(req)
    assert "error" in res
    assert res["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_handle_rpc_request_error_in_tool(exporter: MCPSkillExporter) -> None:
    """Tests error handling when the underlying tool raises an exception."""
    req = {
        "jsonrpc": "2.0",
        "method": "dummy_skill",
        "params": {}, # Missing required param
        "id": 4
    }
    res: Dict[str, Any] = await exporter.handle_rpc_request(req)
    # The adapter wraps it, but the exporter now properly returns a JSON-RPC error.
    assert "error" in res
    assert res["error"]["code"] == -32000
    assert res["error"]["message"].startswith("Error executing skill")
