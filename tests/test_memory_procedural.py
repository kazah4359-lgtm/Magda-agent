import pytest
from magda_agent.memory.procedural import ProceduralMemory

@pytest.fixture
def procedural_memory(tmp_path):
    """
    Fixture to provide an ephemeral ProceduralMemory instance for testing.
    We use ':memory:' to avoid persisting test data to disk.
    """
    return ProceduralMemory(persist_directory=":memory:")

def test_store_and_recall_procedure(procedural_memory: ProceduralMemory) -> None:
    """Test that procedures can be stored and recalled."""
    procedural_memory.store_procedure("test_proc", "Do this, then do that", user_id=1)

    # We expect some match if we query something similar
    results = procedural_memory.recall_procedure("How to test proc?", user_id=1)

    assert len(results) > 0
    assert "Procedure Name: test_proc" in results[0]
    assert "Do this, then do that" in results[0]

def test_procedural_memory_user_context(procedural_memory: ProceduralMemory) -> None:
    """Test that procedural memory isolates records correctly by user ID."""
    procedural_memory.store_procedure("user1_proc", "User 1 secret procedure", user_id=1)
    procedural_memory.store_procedure("user2_proc", "User 2 secret procedure", user_id=2)

    results_user1 = procedural_memory.recall_procedure("secret procedure", user_id=1)
    results_user2 = procedural_memory.recall_procedure("secret procedure", user_id=2)
    results_all = procedural_memory.recall_procedure("secret procedure", user_id=None)

    # Check user 1's recall
    assert any("user1_proc" in r for r in results_user1)
    assert not any("user2_proc" in r for r in results_user1)

    # Check user 2's recall
    assert any("user2_proc" in r for r in results_user2)
    assert not any("user1_proc" in r for r in results_user2)

    # Calling recall without user_id might return both, depending on Chromadb matching,
    # but at the very least it shouldn't fail and should return something.
    assert len(results_all) > 0

def test_store_procedure_with_metadata(procedural_memory: ProceduralMemory) -> None:
    """Test storing procedure with additional metadata."""
    metadata = {"category": "system", "version": "1.0"}
    procedural_memory.store_procedure("meta_proc", "Steps with metadata", metadata=metadata, user_id=1)

    results = procedural_memory.recall_procedure("meta_proc", user_id=1)
    assert len(results) > 0
    assert "meta_proc" in results[0]
