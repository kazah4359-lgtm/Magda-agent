from unittest.mock import AsyncMock
import pytest
from magda_agent.memory.virtual_context_v2 import VirtualContextManagerV2
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState
import asyncio

@pytest.mark.asyncio
async def test_virtual_context_v2_page_out_explicit() -> None:
    wm = WorkingMemory(limit=5)
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_v2_out"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)
    vcm = VirtualContextManagerV2()

    state = PADState(0, 0, 0)

    e1 = MemoryEntry("First Item", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second Item", 0.6, state, user_id=1)
    e3 = MemoryEntry("Third Item", 0.7, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)
    await wm.add(e3)

    assert len(wm.get_entries(user_id=1)) == 3

    # Explicitly page out the first 2 items
    await vcm.page_out_explicit(wm, em, user_id=1, count=2)

    entries = wm.get_entries(user_id=1)
    assert len(entries) == 1
    assert entries[0].content == "Third Item"

    # Verify Episodic Memory has the paged out entry/entries
    episodic_events = em.get_all_events(user_id=1)
    assert len(episodic_events) > 0
    # The paged out metadata should indicate explicit page out
    assert episodic_events[0]["metadata"]["paged_out_explicitly"] == True

@pytest.mark.asyncio
async def test_virtual_context_v2_page_in_explicit() -> None:
    wm = WorkingMemory(limit=5)
    em = EpisodicMemory(persist_directory=":memory:")
    em.collection_name = "test_episodic_memory_v2_in"
    em.collection = em.client.get_or_create_collection(name=em.collection_name)
    vcm = VirtualContextManagerV2()

    em.store_event("Historical facts about Python explicitly", metadata={"paged_out_explicitly": True}, user_id=2)

    await vcm.page_in_explicit(wm, em, user_id=2, query="Python facts", top_k=1)

    entries = wm.get_entries(user_id=2)
    assert len(entries) == 1
    assert "Historical facts about Python explicitly" in entries[0].content

@pytest.mark.asyncio
async def test_virtual_context_v2_compress_without_llm() -> None:
    """Test that context compression works with fallback summary when LLM is absent."""
    vcm = VirtualContextManagerV2()
    state = PADState(0.1, 0.2, 0.3)
    e1 = MemoryEntry("First", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second", 0.5, state, user_id=1)

    summary = await vcm.compress_context([e1, e2])
    assert "Summary of 2 items: First\nSecond" in summary.content
    assert summary.importance == 0.5
    assert summary.user_id == 1

@pytest.mark.asyncio
async def test_virtual_context_v2_compress_with_llm() -> None:
    """Test that context compression leverages the LLM client to generate summaries."""
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "Mocked Summary V2"
    vcm = VirtualContextManagerV2(llm_client=mock_llm)

    state = PADState(0.1, 0.2, 0.3)
    e1 = MemoryEntry("First", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second", 0.5, state, user_id=1)

    summary = await vcm.compress_context([e1, e2])
    assert summary.content == "Mocked Summary V2"
    assert summary.importance == 0.5
    assert summary.user_id == 1
