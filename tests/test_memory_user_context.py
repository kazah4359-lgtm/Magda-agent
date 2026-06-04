import pytest
from magda_agent.memory.long_term import LongTermMemory
from magda_agent.memory.storage import MemorySystem, MemoryEntry
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.engine import PADState

@pytest.fixture
def memory_system():
    return MemorySystem()

@pytest.fixture
def long_term_memory(tmp_path):
    persist_dir = str(tmp_path / "test_long_term_user_db")
    return LongTermMemory(persist_directory=persist_dir)

@pytest.fixture
def habit_tracker(tmp_path):
    persist_dir = str(tmp_path / "test_habits_user_db")
    return HabitTracker(persist_directory=persist_dir)

def test_long_term_memory_user_context(long_term_memory: LongTermMemory) -> None:
    """Test that long-term memory isolates records correctly by user ID."""
    long_term_memory.store("User 1 secret", user_id=1)
    long_term_memory.store("User 2 secret", user_id=2)

    results_user1 = long_term_memory.recall("secret", user_id=1)
    results_user2 = long_term_memory.recall("secret", user_id=2)
    results_all = long_term_memory.recall("secret", user_id=None)

    assert "User 1 secret" in results_user1
    assert "User 2 secret" not in results_user1

    assert "User 2 secret" in results_user2
    assert "User 1 secret" not in results_user2

    # Assuming 'recall' without user_id returns results globally (or based on similarity)
    # the returned result will be the top match, but we know it's storing them separately.

def test_memory_system_user_context(memory_system: MemorySystem) -> None:
    """Test that the short/long-term memory system isolates records correctly by user ID."""
    state = PADState(0.5, 0.5, 0.5)
    memory_system.add_memory("Hello from user 1", importance=0.8, emotional_state=state, user_id=1)
    memory_system.add_memory("Hello from user 2", importance=0.8, emotional_state=state, user_id=2)

    results_user1 = memory_system.retrieve_relevant("Hello", user_id=1)
    results_user2 = memory_system.retrieve_relevant("Hello", user_id=2)

    assert len(results_user1) == 1
    assert results_user1[0].content == "Hello from user 1"

    assert len(results_user2) == 1
    assert results_user2[0].content == "Hello from user 2"

def test_habit_tracker_user_context(habit_tracker: HabitTracker) -> None:
    """Test that habit tracking isolates records correctly by user ID."""
    # Train user 1 habit
    habit_tracker.record_usage("do the task", "skill_A", 9.0, user_id=1)
    habit_tracker.record_usage("do the task", "skill_A", 9.0, user_id=1)

    # Train user 2 habit
    habit_tracker.record_usage("do the task", "skill_B", 9.0, user_id=2)
    habit_tracker.record_usage("do the task", "skill_B", 9.0, user_id=2)

    assert habit_tracker.suggest_strategy("do the task", user_id=1) == "skill_A"
    assert habit_tracker.suggest_strategy("do the task", user_id=2) == "skill_B"

def test_memory_system_vector_search(memory_system: MemorySystem) -> None:
    """Test that the short/long-term memory system retrieves records using semantic vector search."""
    state = PADState(0.5, 0.5, 0.5)

    # Store some contextually distinct memories
    memory_system.add_memory("The quick brown fox jumps over the lazy dog", importance=0.8, emotional_state=state, user_id=1)
    memory_system.add_memory("I love eating apples and bananas", importance=0.6, emotional_state=state, user_id=1)
    memory_system.add_memory("Docker is a containerization technology", importance=0.9, emotional_state=state, user_id=1)

    # Search for an animal concept
    results = memory_system.retrieve_relevant("a fast wild canine", limit=1, user_id=1)
    assert len(results) == 1
    assert "fox" in results[0].content.lower()

    # Search for food concept
    results_food = memory_system.retrieve_relevant("delicious fruits", limit=1, user_id=1)
    assert len(results_food) == 1
    assert "apples" in results_food[0].content.lower()

    # Search for tech concept
    results_tech = memory_system.retrieve_relevant("software deployment", limit=1, user_id=1)
    assert len(results_tech) == 1
    assert "docker" in results_tech[0].content.lower()
