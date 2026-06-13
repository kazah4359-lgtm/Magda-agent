import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.compressor import ContextCompressor
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Compressed summary.")
    return llm

@pytest.fixture
def compressor(mock_llm):
    return ContextCompressor(llm=mock_llm)

@pytest.mark.asyncio
async def test_compress_text_within_limits(compressor, mock_llm):
    text = "Short text"
    result = await compressor.compress_text(text, max_tokens=100)
    assert result == "Short text"
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_text_exceeds_limits(compressor, mock_llm):
    text = "A" * 5000
    result = await compressor.compress_text(text, max_tokens=100)
    assert result == "Compressed summary."
    mock_llm.chat_completion.assert_called_once()

def test_filter_low_salience(compressor):
    pad = PADState(pleasure=0.0, arousal=0.0, dominance=0.0)
    mem1 = MemoryEntry(content="High", importance=0.8, emotional_state=pad)
    mem2 = MemoryEntry(content="Low", importance=0.2, emotional_state=pad)

    filtered = compressor.filter_low_salience([mem1, mem2], threshold=0.5)
    assert len(filtered) == 1
    assert filtered[0].content == "High"

@pytest.mark.asyncio
async def test_compress_memories_filters_and_compresses(compressor, mock_llm):
    pad = PADState(pleasure=0.0, arousal=0.0, dominance=0.0)
    mem1 = MemoryEntry(content="A" * 2500, importance=0.8, emotional_state=pad)
    mem2 = MemoryEntry(content="Low", importance=0.2, emotional_state=pad)
    mem3 = MemoryEntry(content="B" * 2500, importance=0.9, emotional_state=pad)

    result = await compressor.compress_memories([mem1, mem2, mem3], max_tokens=100, threshold=0.5)

    assert len(result) == 1
    assert result[0].content == "Compressed summary."
    assert result[0].importance == pytest.approx(0.85)
    mock_llm.chat_completion.assert_called_once()
