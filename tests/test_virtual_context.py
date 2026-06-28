from unittest.mock import AsyncMock
import pytest
from magda_agent.memory.virtual_context import VirtualContextManager, CoreMemory
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

@pytest.mark.asyncio
async def test_virtual_context_empty_entries() -> None:
    """Test that compressing empty entries raises ValueError."""
    vcm = VirtualContextManager()
    with pytest.raises(ValueError):
        await vcm.compress_context([])

@pytest.mark.asyncio
async def test_virtual_context_compress_without_llm() -> None:
    """Test that context compression works with fallback summary when LLM is absent."""
    vcm = VirtualContextManager()
    state = PADState(0.1, 0.2, 0.3)
    e1 = MemoryEntry("First", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second", 0.5, state, user_id=1)

    summary = await vcm.compress_context([e1, e2])
    assert "Summary of 2 items: First\nSecond" in summary.content
    assert summary.importance == 0.5
    assert summary.user_id == 1

@pytest.mark.asyncio
async def test_virtual_context_compress_with_llm() -> None:
    """Test that context compression leverages the LLM client to generate summaries."""
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "Mocked Summary"
    vcm = VirtualContextManager(llm_client=mock_llm)

    state = PADState(0.1, 0.2, 0.3)
    e1 = MemoryEntry("First", 0.5, state, user_id=1)
    e2 = MemoryEntry("Second", 0.5, state, user_id=1)

    summary = await vcm.compress_context([e1, e2])
    assert summary.content == "Mocked Summary"
    assert summary.importance == 0.5
    assert summary.user_id == 1

@pytest.mark.asyncio
async def test_core_memory_management():
    vcm = VirtualContextManager()
    user_id = 42

    await vcm.update_core_section(user_id, "persona", "I am a helpful assistant.")
    await vcm.update_core_section(user_id, "human", "The user is an engineer.")
    await vcm.update_core_section(user_id, "task", "Implement context management.")

    core = vcm.get_core_memory(user_id)
    assert core.persona == "I am a helpful assistant."
    assert core.human == "The user is an engineer."
    assert core.task == "Implement context management."

    assembled = core.assemble()
    assert "CORE MEMORY (PERSONA):" in assembled
    assert "I am a helpful assistant." in assembled
    assert "CORE MEMORY (HUMAN):" in assembled
    assert "The user is an engineer." in assembled
    assert "CORE MEMORY (TASK):" in assembled
    assert "Implement context management." in assembled

@pytest.mark.asyncio
async def test_core_memory_overflow_truncation():
    # Set a small limit for testing truncation
    vcm = VirtualContextManager(section_limit=5)
    user_id = 123

    long_content = "This is a very long string that will definitely exceed the five word limit."
    await vcm.update_core_section(user_id, "persona", long_content)

    core = vcm.get_core_memory(user_id)
    # Heuristic limit is 5 words
    words = core.persona.split()
    assert len(words) == 5 # limit (last word includes "...")
    assert core.persona.endswith("...")

@pytest.mark.asyncio
async def test_core_memory_overflow_llm_summarization():
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "Summarized Persona"
    vcm = VirtualContextManager(llm_client=mock_llm, section_limit=5)
    user_id = 123

    long_content = "This is a very long string that will definitely exceed the five word limit."
    await vcm.update_core_section(user_id, "persona", long_content)

    core = vcm.get_core_memory(user_id)
    assert core.persona == "Summarized Persona"
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_maintain_working_memory_limits_pages_out():
    wm = WorkingMemory(limit=10) # limit by entries is high
    em = EpisodicMemory(persist_directory=":memory:")
    vcm = VirtualContextManager()

    wm.virtual_context_manager = vcm
    wm.episodic_memory = em

    state = PADState(0, 0, 0)

    # Each entry has 4 words -> ~5 tokens. 3 entries = ~15 tokens
    e1 = MemoryEntry("word1 word2 word3 word4", 0.5, state, user_id=1)
    e2 = MemoryEntry("word5 word6 word7 word8", 0.6, state, user_id=1)
    e3 = MemoryEntry("word9 word10 word11 word12", 0.7, state, user_id=1)

    await wm.add(e1)
    await wm.add(e2)
    await wm.add(e3)

    assert len(wm.get_entries(user_id=1)) == 3
    assert vcm.get_token_length(wm.get_entries(user_id=1)) == int(12 * 1.3)

    # Maintain with max_tokens = 10. Current is ~15, so it should page out entries.
    await vcm.maintain_working_memory_limits(wm, em, user_id=1, max_tokens=10)

    # It pages out max(1, len(entries)//2) = max(1, 1) = 1 entry.
    entries = wm.get_entries(user_id=1)
    assert len(entries) == 2
    assert entries[0].content == "word5 word6 word7 word8"
    assert entries[1].content == "word9 word10 word11 word12"

@pytest.mark.asyncio
async def test_maintain_working_memory_limits_no_page_out():
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")
    vcm = VirtualContextManager()

    wm.virtual_context_manager = vcm
    wm.episodic_memory = em

    state = PADState(0, 0, 0)

    # 4 words -> ~5 tokens
    e1 = MemoryEntry("word1 word2 word3 word4", 0.5, state, user_id=1)
    await wm.add(e1)

    assert len(wm.get_entries(user_id=1)) == 1

    # Limit is 10, current is 5 -> no page out
    await vcm.maintain_working_memory_limits(wm, em, user_id=1, max_tokens=10)

    entries = wm.get_entries(user_id=1)
    assert len(entries) == 1
