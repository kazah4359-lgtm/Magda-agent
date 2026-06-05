import pytest
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.emotions.engine import PADState

def test_working_memory_add_and_get():
    wm = WorkingMemory(limit=2)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("First", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second", 0.6, state, user_id=1)
    e3 = MemoryEntry("Third", 0.7, state, user_id=1)

    wm.add(e1)
    assert len(wm.get_entries(user_id=1)) == 1

    wm.add(e2)
    assert len(wm.get_entries(user_id=1)) == 2

    # Should evict e1
    wm.add(e3)
    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2
    assert entries[0].content == "Second"
    assert entries[1].content == "Third"

def test_working_memory_user_isolation():
    wm = WorkingMemory(limit=5)
    state = PADState(0, 0, 0)

    wm.add(MemoryEntry("User 1", 0.5, state, user_id=1))
    wm.add(MemoryEntry("User 2", 0.5, state, user_id=2))
    wm.add(MemoryEntry("Anon", 0.5, state, user_id=None))

    assert len(wm.get_entries(user_id=1)) == 1
    assert len(wm.get_entries(user_id=2)) == 1
    assert len(wm.get_entries(user_id=None)) == 1

def test_working_memory_remove_and_clear():
    wm = WorkingMemory(limit=5)
    state = PADState(0, 0, 0)

    import time
    e1 = MemoryEntry("A", 0.5, state, user_id=1)
    e1.id = 1
    time.sleep(0.01) # to ensure different ID if relying on time, though we override
    e2 = MemoryEntry("B", 0.5, state, user_id=1)
    e2.id = 2

    wm.add(e1)
    wm.add(e2)

    wm.remove(e1.id, user_id=1)
    assert len(wm.get_entries(user_id=1)) == 1
    assert wm.get_entries(user_id=1)[0].content == "B"

    wm.clear(user_id=1)
    assert len(wm.get_entries(user_id=1)) == 0
