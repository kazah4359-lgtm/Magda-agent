import pytest
from magda_agent.memory.virtual_context import VirtualContextManager
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState
import asyncio

@pytest.mark.asyncio
async def test_virtual_context_page_out():
    wm = WorkingMemory(limit=2)
    em = EpisodicMemory(persist_directory=":memory:")
    vcm = VirtualContextManager()

    wm.virtual_context_manager = vcm
    wm.episodic_memory = em

    state = PADState(0, 0, 0)

    e1 = MemoryEntry("First Item", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second Item", 0.6, state, user_id=1)
    e3 = MemoryEntry("Third Item", 0.7, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)

    assert len(wm.get_entries(user_id=1)) == 2

    # Adding third item should trigger page_out of the oldest (First Item)
    await wm.add(e3)

    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2
    assert entries[0].content == "Second Item"
    assert entries[1].content == "Third Item"

    # Verify Episodic Memory has the paged out entry
    episodic_events = em.get_all_events(user_id=1)
    assert len(episodic_events) == 1
    assert episodic_events[0]["text"] == "First Item"
    assert episodic_events[0]["metadata"]["paged_out"] == True

@pytest.mark.asyncio
async def test_virtual_context_page_in():
    wm = WorkingMemory(limit=5)
    em = EpisodicMemory(persist_directory=":memory:")
    vcm = VirtualContextManager()

    wm.virtual_context_manager = vcm
    wm.episodic_memory = em

    em.store_event("Historical facts about Python", metadata={"paged_out": True}, user_id=2)

    await vcm.page_in(wm, em, user_id=2, query="Python facts")

    entries = wm.get_entries(user_id=2)
    assert len(entries) == 1
    assert "Historical facts about Python" in entries[0].content
