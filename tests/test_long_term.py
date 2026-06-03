import pytest
import os
import shutil
from magda_agent.memory.long_term import LongTermMemory

@pytest.fixture
def temp_memory(tmp_path):
    """Fixture to create and clean up a temporary LongTermMemory instance."""
    persist_dir = str(tmp_path / "test_long_term_db")
    memory = LongTermMemory(persist_directory=persist_dir)
    yield memory

def test_store_and_recall(temp_memory):
    """Test storing a memory and recalling it."""
    # Store a memory
    memory_text = "The quick brown fox jumps over the lazy dog."
    metadata = {"source": "test"}

    temp_memory.store(text=memory_text, metadata=metadata)

    # Recall the memory with a similar semantic query
    query = "fox jumping over dog"
    results = temp_memory.recall(query=query, top_k=1)

    assert len(results) > 0
    assert memory_text in results

def test_recall_empty(temp_memory):
    """Test recalling from an empty memory."""
    query = "something random"
    results = temp_memory.recall(query=query, top_k=1)

    assert len(results) == 0

def test_store_multiple_and_recall_top_k(temp_memory):
    """Test storing multiple memories and recalling top k."""
    memories = [
        "Python is a programming language.",
        "The sky is blue.",
        "Docker is used for containerization."
    ]

    for mem in memories:
        temp_memory.store(mem)

    query = "What is the color of the sky?"
    results = temp_memory.recall(query=query, top_k=2)

    assert len(results) > 0
    assert "The sky is blue." in results
