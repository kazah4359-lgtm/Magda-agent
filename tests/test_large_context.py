import pytest
from magda_agent.memory.large_context import LargeContextWindow

def test_large_context_initialization() -> None:
    """Test that the LargeContextWindow initializes correctly."""
    window = LargeContextWindow(max_tokens=1000000)
    assert window.max_tokens == 1000000
    assert window.current_tokens == 0
    assert len(window.chunks) == 0

def test_large_context_add_chunk() -> None:
    """Test adding a chunk to the LargeContextWindow."""
    window = LargeContextWindow()
    window.add_chunk("This is a test content chunk.", tokens=10)
    assert window.current_tokens == 10
    assert len(window.chunks) == 1
    assert window.chunks[0]["content"] == "This is a test content chunk."

def test_large_context_retrieve() -> None:
    """Test retrieving chunks from the LargeContextWindow."""
    window = LargeContextWindow()
    window.add_chunk("The quick brown fox", tokens=4)
    window.add_chunk("jumps over the lazy dog", tokens=5)
    window.add_chunk("This is unrelated text", tokens=4)

    results = window.retrieve("fox")
    assert len(results) == 1
    assert "fox" in results[0]["content"]

    results = window.retrieve("jumps")
    assert len(results) == 1
    assert "lazy dog" in results[0]["content"]

    results = window.retrieve("nonexistent")
    assert len(results) == 0
