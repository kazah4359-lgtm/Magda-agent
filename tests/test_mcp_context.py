import pytest
from magda_agent.integration.mcp_context import MCPContextExporter
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory
from unittest.mock import MagicMock

def test_mcp_context_exporter_initialization() -> None:
    """Tests the initialization of the MCPContextExporter."""
    mock_engine = MagicMock(spec=ContextEngine)
    exporter = MCPContextExporter(mock_engine)
    assert exporter.context_engine == mock_engine

def test_list_tools() -> None:
    """Tests that list_tools returns the correct MCP tool definition."""
    mock_engine = MagicMock(spec=ContextEngine)
    exporter = MCPContextExporter(mock_engine)
    tools = exporter.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "get_context"

def test_get_context() -> None:
    """Tests retrieving context without backing memories."""
    mock_engine = MagicMock(spec=ContextEngine)
    mock_engine.retrieve_context.return_value = [{"mock": "data"}]
    exporter = MCPContextExporter(mock_engine)
    result = exporter.get_context("test query", 123)
    assert result == [{"mock": "data"}]
    mock_engine.retrieve_context.assert_called_once()

def test_get_context_with_memories() -> None:
    """Tests retrieving context correctly integrates with backing episodic and semantic memory."""
    mock_engine = MagicMock(spec=ContextEngine)

    def side_effect(query: str, user_id: int, base_retrieval_func: callable) -> list:
        return base_retrieval_func(query, user_id)

    mock_engine.retrieve_context.side_effect = side_effect

    mock_episodic = MagicMock(spec=EpisodicMemory)
    mock_episodic.recall_events.return_value = ["event1"]

    mock_semantic = MagicMock(spec=SemanticMemory)
    mock_semantic.recall_facts.return_value = ["concept1"]

    exporter = MCPContextExporter(mock_engine, episodic_memory=mock_episodic, semantic_memory=mock_semantic)

    result = exporter.get_context("test query", 123)
    assert result == ["event1", "concept1"]
    mock_episodic.recall_events.assert_called_once_with(query="test query", top_k=5, user_id=123)
    mock_semantic.recall_facts.assert_called_once_with(query="test query", top_k=5, user_id=123)
