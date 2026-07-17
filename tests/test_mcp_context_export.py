import pytest
import json
from unittest.mock import MagicMock
from magda_agent.skills.mcp_context_export import MCPContextExportTool
from typing import Any, Dict, List, Optional

class MockMemoryEntry:
    """Mock implementation of a memory entry for testing."""
    def __init__(self, id: str, content: str, importance: float, timestamp: float, tags: List[str], user_id: int) -> None:
        """Initialize mock memory entry attributes."""
        self.id = id
        self.content = content
        self.importance = importance
        self.timestamp = timestamp
        self.tags = tags
        self.user_id = user_id

@pytest.fixture
def mock_memory_source() -> MagicMock:
    """Fixture providing a mock memory source representing a ContextEngine."""
    source = MagicMock()

    # Mock data
    entries = [
        MockMemoryEntry("1", "Hello World", 0.9, 1234567890.0, ["greeting"], 42),
        MockMemoryEntry("2", "Test Entry", 0.5, 1234567891.0, ["test"], 42)
    ]

    def get_entries(user_id: Optional[int] = None) -> List[MockMemoryEntry]:
        """Mock get_entries method filtering by user_id if provided."""
        if user_id == 42:
            return entries
        elif user_id is None:
            return entries
        return []

    def get_all_entries() -> List[MockMemoryEntry]:
        """Mock get_all_entries method returning all entries."""
        return entries

    source.get_entries = get_entries
    source.get_all_entries = get_all_entries

    return source

def test_list_tools(mock_memory_source: MagicMock) -> None:
    """Test that the tool correctly lists itself with the proper schema."""
    tool = MCPContextExportTool(mock_memory_source)
    tools = tool.list_tools()

    assert len(tools) == 1
    assert tools[0]["name"] == "export_context_state"
    assert "user_id" in tools[0]["inputSchema"]["properties"]

def test_call_tool_valid(mock_memory_source: MagicMock) -> None:
    """Test calling the tool with a valid tool name and arguments."""
    tool = MCPContextExportTool(mock_memory_source)
    result = tool.call_tool("export_context_state", {"user_id": 42})

    assert not result["isError"]

    # Parse the JSON payload inside the content
    payload = json.loads(result["content"][0]["text"])

    assert len(payload) == 2
    assert payload[0]["id"] == "1"
    assert payload[0]["content"] == "Hello World"
    assert payload[0]["importance"] == 0.9
    assert payload[0]["user_id"] == 42

def test_call_tool_invalid_name(mock_memory_source: MagicMock) -> None:
    """Test calling the tool with an invalid name returns an error format."""
    tool = MCPContextExportTool(mock_memory_source)
    result = tool.call_tool("unknown_tool", {})

    assert result["isError"]
    assert "not found" in result["content"][0]["text"]

def test_call_tool_all_entries(mock_memory_source: MagicMock) -> None:
    """Test calling the tool without filtering arguments exports all entries."""
    tool = MCPContextExportTool(mock_memory_source)
    result = tool.call_tool("export_context_state", {})

    assert not result["isError"]
    payload = json.loads(result["content"][0]["text"])
    assert len(payload) == 2

@pytest.mark.asyncio
async def test_call_tool_async(mock_memory_source: MagicMock) -> None:
    """Test calling the tool async correctly wraps the synchronous call."""
    tool = MCPContextExportTool(mock_memory_source)
    result = await tool.call_tool_async("export_context_state", {"user_id": 42})

    assert not result["isError"]
    payload = json.loads(result["content"][0]["text"])
    assert len(payload) == 2
