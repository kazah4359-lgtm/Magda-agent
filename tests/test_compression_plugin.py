import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.compression_plugin import CompressionPlugin
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Compressed summary")
    return llm

@pytest.fixture
def pad_state():
    return PADState(pleasure=0.0, arousal=0.0, dominance=0.0)

@pytest.mark.asyncio
async def test_compression_plugin_compact_under_limit(mock_llm, pad_state):
    plugin = CompressionPlugin(llm=mock_llm)

    # 2 items, limit 3 -> should not compact
    items = [
        MemoryEntry(content="Item 1", user_id=1, importance=0.5, emotional_state=pad_state),
        MemoryEntry(content="Item 2", user_id=1, importance=0.5, emotional_state=pad_state)
    ]
    metadata = {"limit": 3}

    result = await plugin.compact(items, metadata)

    assert len(result) == 2
    assert result == items
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compression_plugin_compact_over_limit(mock_llm, pad_state):
    plugin = CompressionPlugin(llm=mock_llm)

    # 4 items, limit 3 -> should compact first two
    items = [
        MemoryEntry(content="Item 1", user_id=1, importance=0.6, emotional_state=pad_state, tags=["t1"]),
        MemoryEntry(content="Item 2", user_id=1, importance=0.8, emotional_state=pad_state, tags=["t2"]),
        MemoryEntry(content="Item 3", user_id=1, importance=0.5, emotional_state=pad_state),
        MemoryEntry(content="Item 4", user_id=1, importance=0.5, emotional_state=pad_state)
    ]
    metadata = {"limit": 3}

    result = await plugin.compact(items, metadata)

    assert len(result) == 3
    assert result[0].content == "Compressed summary"
    assert result[0].importance == 0.7  # (0.6 + 0.8) / 2
    assert set(result[0].tags) == {"t1", "t2"}
    assert result[1] == items[2]
    assert result[2] == items[3]
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_compression_plugin_compact_no_llm(pad_state):
    plugin = CompressionPlugin(llm=None)

    items = [
        MemoryEntry(content="Item 1", user_id=1, importance=0.5, emotional_state=pad_state),
        MemoryEntry(content="Item 2", user_id=1, importance=0.5, emotional_state=pad_state),
        MemoryEntry(content="Item 3", user_id=1, importance=0.5, emotional_state=pad_state),
        MemoryEntry(content="Item 4", user_id=1, importance=0.5, emotional_state=pad_state)
    ]
    metadata = {"limit": 3}

    # Should drop oldest item
    result = await plugin.compact(items, metadata)

    assert len(result) == 3
    assert result == items[1:]
