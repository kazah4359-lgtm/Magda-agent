import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, call
from magda_agent.memory.semantic_compaction_v4 import SemanticCompactionV4, register_compaction_cron

@pytest_asyncio.fixture
async def compactor_setup():
    episodic_mock = MagicMock()
    semantic_mock = MagicMock()
    llm_mock = MagicMock()
    llm_mock.generate = AsyncMock()

    compactor = SemanticCompactionV4(
        episodic_memory=episodic_mock,
        semantic_memory=semantic_mock,
        llm_client=llm_mock,
        batch_size=2
    )
    return compactor, episodic_mock, semantic_mock, llm_mock

@pytest.mark.asyncio
async def test_compaction_not_enough_events(compactor_setup):
    compactor, episodic_mock, _, llm_mock = compactor_setup

    # Return fewer events than batch_size (2)
    episodic_mock.get_all_events.return_value = [{"id": "1", "text": "event1", "metadata": {}}]

    await compactor.run_compaction()

    episodic_mock.get_all_events.assert_called_once()
    llm_mock.generate.assert_not_called()

@pytest.mark.asyncio
async def test_compaction_success(compactor_setup):
    compactor, episodic_mock, semantic_mock, llm_mock = compactor_setup

    # Return enough events
    events = [
        {"id": "1", "text": "event1", "metadata": {}},
        {"id": "2", "text": "event2", "metadata": {}},
        {"id": "3", "text": "event3", "metadata": {}}
    ]
    episodic_mock.get_all_events.return_value = events

    # Mock LLM response
    llm_mock.generate.return_value = "- Fact 1\n- Fact 2"

    await compactor.run_compaction()

    # Check LLM called
    llm_mock.generate.assert_called_once()

    # Check facts stored
    assert semantic_mock.store_fact.call_count == 2
    semantic_mock.store_fact.assert_any_call("Fact 1", metadata={"source": "compaction_v4"})
    semantic_mock.store_fact.assert_any_call("Fact 2", metadata={"source": "compaction_v4"})

    # Check original events decayed (only batch_size=2 events should be decayed)
    assert episodic_mock.decay_event.call_count == 2
    episodic_mock.decay_event.assert_any_call("1")
    episodic_mock.decay_event.assert_any_call("2")

@pytest.mark.asyncio
async def test_compaction_llm_error(compactor_setup):
    compactor, episodic_mock, semantic_mock, llm_mock = compactor_setup

    events = [
        {"id": "1", "text": "event1", "metadata": {}},
        {"id": "2", "text": "event2", "metadata": {}},
    ]
    episodic_mock.get_all_events.return_value = events

    llm_mock.generate.return_value = "Error: Some API issue"

    await compactor.run_compaction()

    llm_mock.generate.assert_called_once()
    semantic_mock.store_fact.assert_not_called()
    episodic_mock.decay_event.assert_not_called()

def test_register_compaction_cron():
    scheduler_mock = MagicMock()
    scheduler_mock.task.return_value = lambda x: x

    episodic_mock = MagicMock()
    semantic_mock = MagicMock()
    llm_mock = MagicMock()

    compactor = register_compaction_cron(scheduler_mock, episodic_mock, semantic_mock, llm_mock)

    assert isinstance(compactor, SemanticCompactionV4)
    scheduler_mock.task.assert_called_once_with("0 2 * * *", name="semantic_compaction_v4")
