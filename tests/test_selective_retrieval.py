import pytest
import asyncio
from unittest.mock import AsyncMock
from magda_agent.memory.selective_retrieval import SelectiveContextCompressor
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_selective_compressor_no_overflow():
    """Test that if the token limit is not breached, original memories are returned."""
    mock_llm = AsyncMock()
    compressor = SelectiveContextCompressor(llm=mock_llm)

    memories = [
        MemoryEntry("A short memory", 0.5, PADState(0, 0, 0), ["tag1"], 1),
        MemoryEntry("Another short memory", 0.6, PADState(0, 0, 0), ["tag2"], 1)
    ]

    # Very high max_tokens so it won't overflow
    result = await compressor.compress_old_memories(memories, max_tokens=1000)

    # Should not call LLM
    mock_llm.chat_completion.assert_not_called()
    assert result == memories
    assert len(result) == 2

@pytest.mark.asyncio
async def test_selective_compressor_with_overflow():
    """Test that if token limit is breached, older memories are summarized and newer are kept."""
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "Compressed older memories."
    compressor = SelectiveContextCompressor(llm=mock_llm)

    # Create 4 memories. Let's say we set max_tokens very low,
    # each word is ~1.3 tokens, so 4 words = 5 tokens. Total here is ~20 tokens.
    memories = [
        MemoryEntry("Old memory one", 0.4, PADState(0.1, 0, 0), ["t1"], 1),
        MemoryEntry("Old memory two", 0.6, PADState(0.3, 0, 0), ["t2"], 1),
        MemoryEntry("New memory one", 0.8, PADState(0.5, 0, 0), ["t3"], 1),
        MemoryEntry("New memory two", 0.9, PADState(0.7, 0, 0), ["t4"], 1)
    ]

    # Max tokens = 10, will trigger compression
    result = await compressor.compress_old_memories(memories, max_tokens=10)

    mock_llm.chat_completion.assert_called_once()
    args, kwargs = mock_llm.chat_completion.call_args
    assert args[0][0]["role"] == "system"
    assert "context compression engine" in args[0][0]["content"]

    # We should have [compressed_summary] + [new_memory_one, new_memory_two]
    assert len(result) == 3
    assert result[0].content == "Compressed older memories."
    assert result[0].importance == 0.5  # Average of 0.4 and 0.6
    assert result[0].emotional_state.pleasure == pytest.approx(0.2) # Average of 0.1 and 0.3
    assert "t1" in result[0].tags
    assert "t2" in result[0].tags

    assert result[1] == memories[2]
    assert result[2] == memories[3]

@pytest.mark.asyncio
async def test_selective_compressor_empty_memories():
    """Test with an empty list."""
    mock_llm = AsyncMock()
    compressor = SelectiveContextCompressor(llm=mock_llm)

    result = await compressor.compress_old_memories([], max_tokens=10)
    assert result == []
    mock_llm.chat_completion.assert_not_called()
