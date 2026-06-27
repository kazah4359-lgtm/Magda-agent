import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magda_agent.agents.compression_subagent import ContextCompressionSubagent

@pytest.fixture
def mock_llm_client():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Compressed summary of events.")
    return llm

@pytest.fixture
def mock_episodic_memory():
    memory = MagicMock()
    # Mock some undecayed events
    memory.get_all_events.return_value = [
        {"id": "1", "text": "Event 1", "metadata": {"user_id": 123}},
        {"id": "2", "text": "Event 2", "metadata": {"user_id": 123}},
    ]
    return memory

@pytest.mark.asyncio
async def test_compress_next_batch_logic(mock_llm_client, mock_episodic_memory):
    subagent = ContextCompressionSubagent(
        llm=mock_llm_client,
        episodic_memory=mock_episodic_memory,
        batch_size=10
    )

    await subagent.compress_next_batch()

    # Verify episodic memory was queried
    mock_episodic_memory.get_all_events.assert_called_once_with(include_decayed=False, limit=10)

    # Verify LLM was called to summarize
    mock_llm_client.chat_completion.assert_awaited_once()

    # Verify new summary was stored
    mock_episodic_memory.store_event.assert_called_once_with(
        text="Compressed summary of events.",
        metadata={"type": "compressed_summary"},
        user_id=123
    )

    # Verify original events were decayed
    assert mock_episodic_memory.decay_event.call_count == 2
    mock_episodic_memory.decay_event.assert_any_call("1")
    mock_episodic_memory.decay_event.assert_any_call("2")

@pytest.mark.asyncio
async def test_compress_next_batch_not_enough_events(mock_llm_client, mock_episodic_memory):
    mock_episodic_memory.get_all_events.return_value = [
        {"id": "1", "text": "Event 1", "metadata": {"user_id": 123}}
    ]

    subagent = ContextCompressionSubagent(
        llm=mock_llm_client,
        episodic_memory=mock_episodic_memory,
        batch_size=10
    )

    await subagent.compress_next_batch()

    # Should not call LLM if < 2 events
    mock_llm_client.chat_completion.assert_not_called()
    mock_episodic_memory.store_event.assert_not_called()
    mock_episodic_memory.decay_event.assert_not_called()

@pytest.mark.asyncio
async def test_compression_loop_runs(mock_llm_client, mock_episodic_memory):
    subagent = ContextCompressionSubagent(
        llm=mock_llm_client,
        episodic_memory=mock_episodic_memory,
        sleep_interval=0.01  # very short sleep for test
    )

    # Mock compress_next_batch to ensure it gets called
    with patch.object(subagent, 'compress_next_batch', new_callable=AsyncMock) as mock_compress:
        # Start the loop
        await subagent.start()

        # Let it run for a short time
        await asyncio.sleep(0.05)

        # Stop the loop
        await subagent.stop()

        # Verify it was called multiple times
        assert mock_compress.call_count > 0
