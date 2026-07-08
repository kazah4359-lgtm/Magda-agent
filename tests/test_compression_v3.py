import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.compression_v3 import OpenClawContextCompressorV3
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.fixture
def mock_llm():
    """Provides a mock LLM client."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Compressed summary of non-critical memories.")
    return llm

@pytest.fixture
def compressor(mock_llm):
    """Provides a configured OpenClawContextCompressorV3 instance."""
    return OpenClawContextCompressorV3(llm=mock_llm, threshold=0.8)

@pytest.mark.asyncio
async def test_compress_and_retrieve_under_limit(compressor, mock_llm) -> None:
    """Tests that entries under the token limit are not compressed."""
    entries = [
        MemoryEntry(content="Small entry related to query", importance=0.5, emotional_state=PADState(0,0,0)),
    ]

    result = await compressor.compress_and_retrieve(entries, query="query", max_tokens=1000)

    assert len(result) == 1
    assert result == entries
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_and_retrieve_critical_kept(compressor, mock_llm) -> None:
    """Tests that critical memories are kept uncompressed while others are compressed."""
    # Create entries that will exceed the max_tokens when combined.
    # We will use a very low max_tokens to force compression.
    entries = [
        MemoryEntry(content="Critical memory about query. " * 50, importance=0.9, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="Non-critical detail 1 about query.", importance=0.5, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="Non-critical detail 2 about query.", importance=0.6, emotional_state=PADState(0,0,0)),
    ]

    # max_tokens=10 means max 40 chars. The combined text will exceed this.
    result = await compressor.compress_and_retrieve(entries, query="query", max_tokens=10)

    assert len(result) == 2

    critical = [m for m in result if m.importance >= 0.8]
    assert len(critical) == 1
    assert critical[0].content.startswith("Critical memory about query.")

    compressed = [m for m in result if m.importance < 0.8]
    assert len(compressed) == 1
    assert compressed[0].content == "Compressed summary of non-critical memories."
    assert compressed[0].importance == 0.55

    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_compress_and_retrieve_no_match(compressor, mock_llm) -> None:
    """Tests that entries not matching the query are filtered out."""
    entries = [
        MemoryEntry(content="This is about apples.", importance=0.5, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="This is about bananas.", importance=0.5, emotional_state=PADState(0,0,0)),
    ]

    result = await compressor.compress_and_retrieve(entries, query="oranges", max_tokens=1000)

    assert len(result) == 0
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_and_retrieve_llm_failure(compressor, mock_llm) -> None:
    """Tests fallback behavior when LLM compression fails."""
    mock_llm.chat_completion.side_effect = Exception("LLM Error")

    entries = [
        MemoryEntry(content="Non-critical detail 1 about query.", importance=0.5, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="Non-critical detail 2 about query.", importance=0.6, emotional_state=PADState(0,0,0)),
    ]

    result = await compressor.compress_and_retrieve(entries, query="query", max_tokens=5)

    assert len(result) == 1
    assert "Non-critical detail 1" in result[0].content
    assert "Non-critical detail 2" in result[0].content
