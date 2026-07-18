import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.subagent_compression import SubagentContextCompressor
from magda_agent.agents.context_compression import RPCPayloadCompressor
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

@pytest.mark.asyncio
async def test_rpc_payload_compressor_short():
    llm_mock = MagicMock(spec=LLMClient)
    compressor = RPCPayloadCompressor(llm=llm_mock)

    payload = {"task": "Do something", "context": "Short context"}
    result = await compressor.compress_payload(payload, max_length=100)

    assert result["context"] == "Short context"
    llm_mock.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_rpc_payload_compressor_long():
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Compressed context summary.")
    compressor = RPCPayloadCompressor(llm=llm_mock)

    payload = {"task": "Do something", "context": "A" * 3000}
    result = await compressor.compress_payload(payload, max_length=2000)

    assert result["context"] == "Compressed context summary."
    llm_mock.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_rpc_payload_compressor_preserves_constraints():
    llm_mock = MagicMock(spec=LLMClient)
    llm_mock.chat_completion = AsyncMock(return_value="Compressed context summary missing constraints.")
    compressor = RPCPayloadCompressor(llm=llm_mock)

    long_context = "A" * 3000 + "\nUser must never be ignored.\nThe goal is to compress."
    payload = {"task": "Do something", "context": long_context}

    result = await compressor.compress_payload(payload, max_length=2000)

    assert "Compressed context summary missing constraints." in result["context"]
    assert "User must never be ignored." in result["context"]
    assert "The goal is to compress." in result["context"]
    llm_mock.chat_completion.assert_called_once()
