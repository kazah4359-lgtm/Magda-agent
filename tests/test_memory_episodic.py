import pytest
from magda_agent.memory.episodic import EpisodicMemory

@pytest.fixture
def episodic_memory(tmp_path):
    persist_dir = str(tmp_path / "test_episodic_db")
    return EpisodicMemory(persist_directory=persist_dir)

def test_episodic_memory_store_and_recall(episodic_memory: EpisodicMemory):
    episodic_memory.store_event("User said hello", user_id=1)
    episodic_memory.store_event("User asked about the weather", user_id=1)

    results = episodic_memory.recall_events("weather", top_k=1, user_id=1)
    assert len(results) == 1
    assert "weather" in results[0].lower()

def test_episodic_memory_user_isolation(episodic_memory: EpisodicMemory):
    episodic_memory.store_event("User 1 secret", user_id=1)
    episodic_memory.store_event("User 2 secret", user_id=2)

    results_user1 = episodic_memory.recall_events("secret", user_id=1)
    results_user2 = episodic_memory.recall_events("secret", user_id=2)

    assert len(results_user1) == 1
    assert "User 1" in results_user1[0]

    assert len(results_user2) == 1
    assert "User 2" in results_user2[0]
