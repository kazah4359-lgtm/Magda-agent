import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.subagent_compression import SubagentContextCompressor
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_compress_context_short():
    """Test that short context is not compressed."""
    llm_mock = MagicMock(spec=LLMClient)
    compressor = SubagentContextCompressor(llm=llm_mock)

    context = "Short context"
    result = await compressor.compress_context(context, max_length=100)

    assert result == "Short context"
    llm_mock.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_context_long():
    """Test that long context is compressed."""
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Compressed context summary.")
    compressor = SubagentContextCompressor(llm=llm_mock)

    context = "A" * 3000
    result = await compressor.compress_context(context, max_length=2000)

    assert result == "Compressed context summary."
    llm_mock.chat_completion.assert_called_once()
