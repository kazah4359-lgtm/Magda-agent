import pytest
import asyncio
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_working_memory_add_and_get():
    wm = WorkingMemory(limit=2)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("First", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second", 0.6, state, user_id=1)
    e3 = MemoryEntry("Third", 0.7, state, user_id=1)

    await wm.add(e1)
    assert len(wm.get_entries(user_id=1)) == 1

    await wm.add(e2)
    assert len(wm.get_entries(user_id=1)) == 2

    # Should evict e1
    await wm.add(e3)
    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2
    assert entries[0].content == "Second"
    assert entries[1].content == "Third"

@pytest.mark.asyncio
async def test_working_memory_user_isolation():
    wm = WorkingMemory(limit=5)
    state = PADState(0, 0, 0)

    await wm.add(MemoryEntry("User 1", 0.5, state, user_id=1))
    await wm.add(MemoryEntry("User 2", 0.5, state, user_id=2))
    await wm.add(MemoryEntry("Anon", 0.5, state, user_id=None))

    assert len(wm.get_entries(user_id=1)) == 1
    assert len(wm.get_entries(user_id=2)) == 1
    assert len(wm.get_entries(user_id=None)) == 1

@pytest.mark.asyncio
async def test_working_memory_remove_and_clear():
    wm = WorkingMemory(limit=5)
    state = PADState(0, 0, 0)

    import time
    e1 = MemoryEntry("A", 0.5, state, user_id=1)
    e1.id = 1
    time.sleep(0.01) # to ensure different ID if relying on time, though we override
    e2 = MemoryEntry("B", 0.5, state, user_id=1)
    e2.id = 2

    await wm.add(e1)
    await wm.add(e2)

    wm.remove(e1.id, user_id=1)
    assert len(wm.get_entries(user_id=1)) == 1
    assert wm.get_entries(user_id=1)[0].content == "B"

    wm.clear(user_id=1)
    assert len(wm.get_entries(user_id=1)) == 0

@pytest.mark.asyncio
async def test_working_memory_summarizer():
    wm = WorkingMemory(limit=2)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("Message 1", 0.5, state, user_id=1)
    e2 = MemoryEntry("Message 2", 0.6, state, user_id=1)
    e3 = MemoryEntry("Message 3", 0.7, state, user_id=1)

    async def mock_summarizer(entries):
        assert len(entries) == 2
        content = "Summary of: " + " and ".join([e.content for e in entries])
        return MemoryEntry(content, 0.55, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)
    # This add will exceed the limit (len will be 3), triggering summarizer on e1 and e2
    await wm.add(e3, summarizer=mock_summarizer)

    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2
    assert entries[0].content == "Summary of: Message 1 and Message 2"
    assert entries[1].content == "Message 3"
