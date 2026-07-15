import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
from magda_agent.llm_client import LLMClient
from magda_agent.agents.token_compression import SubagentTokenCompressor
from magda_agent.agents.sub_agent import SubAgent

@pytest.fixture
def mock_llm() -> MagicMock:
    """
    Creates and returns a mocked LLM client.
    """
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock(return_value="Compressed summary content.")
    return llm

@pytest.mark.asyncio
async def test_compress_context_short(mock_llm: MagicMock) -> None:
    """
    Tests that context is returned as-is when it does not exceed max_length.
    """
    compressor = SubagentTokenCompressor(llm=mock_llm)
    context = "Small context."
    task = "Simple task."

    result = await compressor.compress_context(context, task, max_length=100)

    expected = f"Parent Context:\n{context}\n\nAssigned Task:\n{task}"
    assert result == expected
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_context_long(mock_llm: MagicMock) -> None:
    """
    Tests that context is compressed using LLM when it exceeds max_length.
    """
    compressor = SubagentTokenCompressor(llm=mock_llm)
    context = "This is a very long context that will exceed fifty characters because it is extremely verbose."
    task = "Extract key action points."

    result = await compressor.compress_context(context, task, max_length=50)

    assert result == "Compressed summary content."
    mock_llm.chat_completion.assert_awaited_once()

@pytest.mark.asyncio
async def test_compress_context_with_critical_constraints(mock_llm: MagicMock) -> None:
    """
    Tests that critical constraints containing keywords like MUST, NEVER are preserved or re-injected.
    """
    # Let's mock LLM returning a summary that omits the MUST constraint
    mock_llm.chat_completion = AsyncMock(return_value="General summary of context.")
    compressor = SubagentTokenCompressor(llm=mock_llm)
    context = "Background details. The agent MUST NOT write to root. Secure files only."
    task = "Process files."

    result = await compressor.compress_context(context, task, max_length=20)

    assert "MUST NOT write to root" in result
    assert "General summary of context." in result

@pytest.mark.asyncio
async def test_compress_messages_short(mock_llm: MagicMock) -> None:
    """
    Tests that message list is returned unchanged if message count is within limit.
    """
    compressor = SubagentTokenCompressor(llm=mock_llm)
    messages = [
        {"role": "system", "content": "Sys prompt"},
        {"role": "user", "content": "Hello"}
    ]
    task = "Answer user hello."

    result = await compressor.compress_messages(messages, task, max_messages=5)

    assert result == messages
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_compress_messages_long(mock_llm: MagicMock) -> None:
    """
    Tests that older messages are summarized and combined with the system prompt and recent messages.
    """
    mock_llm.chat_completion = AsyncMock(return_value="Summarized discussion on files.")
    compressor = SubagentTokenCompressor(llm=mock_llm)
    messages = [
        {"role": "system", "content": "Sys prompt"},
        {"role": "user", "content": "Msg 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Msg 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Msg 3"},
        {"role": "assistant", "content": "Reply 3"}
    ]
    task = "Complete file processing."

    # We set max_messages to 2, so older messages should be compressed.
    result = await compressor.compress_messages(messages, task, max_messages=2)

    # Check that system prompt is kept
    assert result[0] == {"role": "system", "content": "Sys prompt"}

    # Check that compressed history message is injected
    assert result[1]["role"] == "system"
    assert "Summarized discussion on files." in result[1]["content"]

    # Check that the last 2 messages are retained
    assert result[-2] == {"role": "user", "content": "Msg 3"}
    assert result[-1] == {"role": "assistant", "content": "Reply 3"}

@pytest.mark.asyncio
async def test_subagent_execution_with_token_compression(mock_llm: MagicMock) -> None:
    """
    Tests SubAgent execution with token compression enabled.
    """
    # Setup LLM response for subagent task execution
    mock_llm.chat_completion.side_effect = [
        "Compressed parent context details.",  # first call: context compression
        "Final task execution response."        # second call: execution response
    ]

    subagent = SubAgent(llm=mock_llm, use_enhanced_compression=True)

    context = "A very long parent context that will be compressed because it exceeds the length of 2000 characters. " * 30
    task = "Summarize the findings."

    result = await subagent.execute(task, context)

    assert result == "Final task execution response."
    assert mock_llm.chat_completion.call_count == 2
