import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List, Any
from magda_agent.memory.context_engine_local_v2 import LocalFirstContextEngineV2
from magda_agent.memory.working import MemoryEntry

from magda_agent.emotions.engine import PADState

@pytest.fixture
def memory_entries() -> List[MemoryEntry]:
    return [
        MemoryEntry(content="Item 1", importance=0.8, emotional_state=PADState(0.0, 0.0, 0.0), user_id=1),
        MemoryEntry(content="Item 2", importance=0.6, emotional_state=PADState(0.0, 0.0, 0.0), user_id=1),
        MemoryEntry(content="Item 3", importance=0.5, emotional_state=PADState(0.0, 0.0, 0.0), user_id=1)
    ]

@pytest.mark.asyncio
async def test_primary_llm_success(memory_entries: List[MemoryEntry]) -> None:
    """Test that primary LLM successfully compacts when limit is exceeded."""
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "Summary of 1 and 2"
    mock_local_llm = AsyncMock()

    engine = LocalFirstContextEngineV2(llm=mock_llm, local_llm=mock_local_llm)
    metadata = {"limit": 2}

    compacted = await engine.compact(memory_entries, metadata)

    assert len(compacted) == 2
    assert compacted[0].content == "Summary of 1 and 2"
    assert compacted[1].content == "Item 3"

    mock_llm.chat_completion.assert_awaited_once()
    mock_local_llm.chat_completion.assert_not_awaited()

@pytest.mark.asyncio
async def test_local_fallback_success(memory_entries: List[MemoryEntry]) -> None:
    """Test that it falls back to local_llm if primary llm fails."""
    mock_llm = AsyncMock()
    mock_llm.chat_completion.side_effect = Exception("Primary API Error")
    mock_local_llm = AsyncMock()
    mock_local_llm.chat_completion.return_value = "Local summary"

    engine = LocalFirstContextEngineV2(llm=mock_llm, local_llm=mock_local_llm)
    metadata = {"limit": 2}

    compacted = await engine.compact(memory_entries, metadata)

    assert len(compacted) == 2
    assert compacted[0].content == "Local summary"
    assert compacted[1].content == "Item 3"

    mock_llm.chat_completion.assert_awaited_once()
    mock_local_llm.chat_completion.assert_awaited_once()

@pytest.mark.asyncio
async def test_both_llms_fail_truncation(memory_entries: List[MemoryEntry]) -> None:
    """Test that it truncates the oldest item if both llms fail."""
    mock_llm = AsyncMock()
    mock_llm.chat_completion.side_effect = Exception("Primary API Error")
    mock_local_llm = AsyncMock()
    mock_local_llm.chat_completion.side_effect = Exception("Local API Error")

    engine = LocalFirstContextEngineV2(llm=mock_llm, local_llm=mock_local_llm)
    metadata = {"limit": 2}

    compacted = await engine.compact(memory_entries, metadata)

    assert len(compacted) == 2
    assert compacted[0].content == "Item 2"
    assert compacted[1].content == "Item 3"

    mock_llm.chat_completion.assert_awaited_once()
    mock_local_llm.chat_completion.assert_awaited_once()

@pytest.mark.asyncio
async def test_no_llms_truncation(memory_entries: List[MemoryEntry]) -> None:
    """Test that it truncates the oldest item if both llms are None."""
    engine = LocalFirstContextEngineV2(llm=None, local_llm=None)
    metadata = {"limit": 2}

    compacted = await engine.compact(memory_entries, metadata)

    assert len(compacted) == 2
    assert compacted[0].content == "Item 2"
    assert compacted[1].content == "Item 3"
