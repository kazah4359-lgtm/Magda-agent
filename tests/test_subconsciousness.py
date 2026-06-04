import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.subconsciousness.reflection import Subconsciousness
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem, MemoryEntry

@pytest.fixture
def mock_llm_client():
    mock = AsyncMock(spec=LLMClient)
    mock.chat_completion.return_value = "I am doing well."
    return mock

@pytest.fixture
def mock_emotions():
    mock = MagicMock(spec=EmotionalEngine)
    mock.state = MagicMock()
    return mock

@pytest.fixture
def mock_memory_system():
    mock = MagicMock(spec=MemorySystem)
    mock.short_term = [MemoryEntry(content="Test content 1", timestamp=1000, importance=0.5, emotional_state=MagicMock())]
    return mock

@pytest.fixture
def subconsciousness(mock_llm_client, mock_emotions, mock_memory_system):
    return Subconsciousness(
        llm=mock_llm_client,
        emotions=mock_emotions,
        memory=mock_memory_system,
        interval=10
    )

@pytest.mark.asyncio
async def test_subconsciousness_instantiation(subconsciousness):
    assert subconsciousness.interval == 10
    assert subconsciousness.is_running is False

@pytest.mark.asyncio
async def test_subconsciousness_reflect(subconsciousness, mock_llm_client, mock_emotions, mock_memory_system):
    await subconsciousness.reflect()

    # Verify memory consolidation was called
    mock_memory_system.consolidate.assert_called_once()

    # Verify LLM was called
    mock_llm_client.chat_completion.assert_called_once()

    # Verify emotions were updated
    mock_emotions.update.assert_called_once_with(0.02, -0.01, 0.05)

    # Verify reflection was stored in memory
    mock_memory_system.add_memory.assert_called_once()
    call_args = mock_memory_system.add_memory.call_args[1]
    assert "Subconscious reflection: I am doing well." in call_args["content"]
    assert call_args["tags"] == ["reflection", "internal"]
    assert call_args["importance"] == 0.4

@pytest.mark.asyncio
async def test_subconsciousness_reflect_no_short_term(subconsciousness, mock_llm_client, mock_memory_system):
    # Empty short term memory
    mock_memory_system.short_term = []

    await subconsciousness.reflect()

    # Verify memory consolidation was NOT called
    mock_memory_system.consolidate.assert_not_called()

    # Verify LLM was NOT called
    mock_llm_client.chat_completion.assert_not_called()
