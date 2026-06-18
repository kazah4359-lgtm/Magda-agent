import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.compression_v2 import ContextCompressorV2
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Compressed summary.")
    return llm

@pytest.fixture
def compressor(mock_llm):
    return ContextCompressorV2(llm=mock_llm)

@pytest.mark.asyncio
async def test_compress_and_retrieve_filters(compressor, mock_llm):
    pad = PADState(pleasure=0.0, arousal=0.0, dominance=0.0)
    mem1 = MemoryEntry(content="We should use python for the project.", importance=0.8, emotional_state=pad)
    mem2 = MemoryEntry(content="We need to buy milk.", importance=0.2, emotional_state=pad)

    result = await compressor.compress_and_retrieve([mem1, mem2], query="python", max_tokens=100)
    assert len(result) == 1
    assert "python" in result[0].content.lower()
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_and_retrieve_compresses(compressor, mock_llm):
    pad = PADState(pleasure=0.0, arousal=0.0, dominance=0.0)
    mem1 = MemoryEntry(content="python is great " * 1000, importance=0.8, emotional_state=pad)

    result = await compressor.compress_and_retrieve([mem1], query="python", max_tokens=10)

    assert len(result) == 1
    assert result[0].content == "Compressed summary."
    mock_llm.chat_completion.assert_called_once()
