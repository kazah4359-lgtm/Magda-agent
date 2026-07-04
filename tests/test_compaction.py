import pytest
import asyncio
from unittest.mock import MagicMock
from magda_agent.memory.compaction import MemoryCompactor

@pytest.fixture
def mock_memory_system():
    ms = MagicMock()
    # Mock working memory users
    ms.working_memory._entries_by_user = {1: [], 2: []}

    # Mock consolidate
    ms.consolidate = MagicMock()

    # Mock episodic memory
    ms.episodic_memory = MagicMock()
    ms.episodic_memory.get_all_events.return_value = [
        {"id": "event_1"}, {"id": "event_2"}, {"id": "event_3"}
    ]
    ms.episodic_memory.decay_event = MagicMock()

    return ms

@pytest.mark.asyncio
async def test_compactor_start_stop(mock_memory_system):
    compactor = MemoryCompactor(mock_memory_system, interval_seconds=1, episodic_limit=2)

    # Assert not running initially
    assert not compactor._running
    assert compactor._task is None

    await compactor.start()
    assert compactor._running
    assert compactor._task is not None
    assert not compactor._task.done()

    await compactor.stop()
    assert not compactor._running
    assert compactor._task.done()

def test_compact_memory_logic(mock_memory_system):
    # Set limit to 2, and mock returns 3 events, so 1 event should decay
    compactor = MemoryCompactor(mock_memory_system, interval_seconds=1, episodic_limit=2)

    compactor.compact_memory()

    # Should consolidate for user 1, user 2, and None
    assert mock_memory_system.consolidate.call_count == 3
    mock_memory_system.consolidate.assert_any_call(user_id=1)
    mock_memory_system.consolidate.assert_any_call(user_id=2)
    mock_memory_system.consolidate.assert_any_call(user_id=None)

    # For each user (1, 2, None), it should get events and decay the oldest one (since 3 > 2)
    assert mock_memory_system.episodic_memory.get_all_events.call_count == 3
    assert mock_memory_system.episodic_memory.decay_event.call_count == 3

    # Assert the specific event was decayed
    mock_memory_system.episodic_memory.decay_event.assert_any_call("event_1")

@pytest.mark.asyncio
async def test_run_background_routine(mock_memory_system):
    compactor = MemoryCompactor(mock_memory_system, interval_seconds=0.1, episodic_limit=2)

    # Mock compact_memory to just count calls and stop the loop after 1 pass
    call_count = 0
    def mock_compact_memory():
        nonlocal call_count
        call_count += 1
        compactor._running = False

    compactor.compact_memory = mock_compact_memory

    compactor._running = True
    await compactor.run_background_routine()

    assert call_count == 1
    assert not compactor._running
