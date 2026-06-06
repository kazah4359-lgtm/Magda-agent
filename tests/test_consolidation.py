import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.subconsciousness.reflection import Subconsciousness
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory

@pytest.fixture
def episodic_memory(tmp_path):
    persist_dir = str(tmp_path / "test_episodic_db")
    return EpisodicMemory(persist_directory=persist_dir)

@pytest.fixture
def semantic_memory(tmp_path):
    persist_dir = str(tmp_path / "test_semantic_db")
    return SemanticMemory(persist_directory=persist_dir)

@pytest.mark.asyncio
async def test_consolidate_episodic_to_semantic(episodic_memory, semantic_memory):
    llm_mock = AsyncMock()
    emotions_mock = MagicMock()
    memory_system_mock = MagicMock()
    memory_system_mock.episodic_memory = episodic_memory

    subconsciousness = Subconsciousness(
        llm=llm_mock,
        emotions=emotions_mock,
        memory=memory_system_mock,
        semantic_memory=semantic_memory
    )

    # Store 5 events to trigger consolidation threshold
    for i in range(5):
        episodic_memory.store_event(f"User test event {i}", user_id=1)

    events = episodic_memory.get_all_events()
    decay_ids = [e['id'] for e in events[:2]]

    mock_response = json.dumps({
        "new_facts": ["User likes to test things"],
        "decay_ids": decay_ids
    })
    llm_mock.chat_completion.return_value = mock_response

    await subconsciousness.consolidate_episodic_to_semantic()

    # Check that the fact was stored
    facts = semantic_memory.search_facts("test", top_k=5)
    assert len(facts) == 1
    assert "User likes to test things" in facts[0]["text"]

    # Check that events were decayed
    all_events = episodic_memory.get_all_events(include_decayed=True)
    decayed_events = [e for e in all_events if e['metadata'].get('decayed') == True]
    assert len(decayed_events) == 2
    for eid in decay_ids:
        assert any(e['id'] == eid for e in decayed_events)
