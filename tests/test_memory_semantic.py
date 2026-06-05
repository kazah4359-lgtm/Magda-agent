import pytest
from magda_agent.memory.semantic import SemanticMemory

@pytest.fixture
def semantic_memory(tmp_path):
    persist_dir = str(tmp_path / "test_semantic_db")
    return SemanticMemory(persist_directory=persist_dir)

def test_semantic_memory_store_and_recall(semantic_memory: SemanticMemory):
    semantic_memory.store_fact("The sky is blue", user_id=1)
    semantic_memory.store_fact("Water is H2O", user_id=1)

    results = semantic_memory.recall_facts("liquid", top_k=1, user_id=1)
    assert len(results) == 1
    assert "H2O" in results[0]

def test_semantic_memory_user_isolation(semantic_memory: SemanticMemory):
    semantic_memory.store_fact("User 1 is named Alice", user_id=1)
    semantic_memory.store_fact("User 2 is named Bob", user_id=2)

    results_user1 = semantic_memory.recall_facts("name", user_id=1)
    results_user2 = semantic_memory.recall_facts("name", user_id=2)

    assert len(results_user1) == 1
    assert "Alice" in results_user1[0]

    assert len(results_user2) == 1
    assert "Bob" in results_user2[0]
