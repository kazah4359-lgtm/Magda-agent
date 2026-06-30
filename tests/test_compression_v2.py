import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from magda_agent.skills.compression_v2 import OpenClawContextCompressorV2
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Summary of memory context.")
    return llm

@pytest.fixture
def compressor(mock_llm):
    return OpenClawContextCompressorV2(llm=mock_llm, threshold=2)

@pytest.fixture
def context_engine(compressor):
    return ContextEngine(plugins=[compressor])

@pytest.mark.asyncio
async def test_compact_logic(compressor, mock_llm):
    # Setup entries over threshold (threshold=2)
    entries = [
        MemoryEntry(content="Entry 1", importance=0.5, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="Entry 2", importance=0.5, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="Entry 3", importance=0.5, emotional_state=PADState(0,0,0)),
    ]

    result = await compressor.compact(entries, {"limit": 2})

    # Should have 2 entries now: [Summary of 1 and 2] + [Entry 3]
    assert len(result) == 2
    assert result[0].content == "Summary of memory context."
    assert result[1].content == "Entry 3"
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_context_engine_integration(context_engine, mock_llm):
    entries = [
        MemoryEntry(content="Entry 1", importance=0.5, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="Entry 2", importance=0.5, emotional_state=PADState(0,0,0)),
        MemoryEntry(content="Entry 3", importance=0.5, emotional_state=PADState(0,0,0)),
    ]

    # Trigger compact via context engine
    result = await context_engine.compact(entries, {"limit": 2})

    assert len(result) == 2
    assert result[0].content == "Summary of memory context."
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_compact_under_limit(compressor, mock_llm):
    entries = [
        MemoryEntry(content="Entry 1", importance=0.5, emotional_state=PADState(0,0,0)),
    ]

    result = await compressor.compact(entries, {"limit": 2})

    assert len(result) == 1
    assert result == entries
    mock_llm.chat_completion.assert_not_called()
