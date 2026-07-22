from unittest.mock import AsyncMock
import pytest
from magda_agent.memory.virtual_context import VirtualContextManager, CoreMemory
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.emotions.engine import PADState
import asyncio

@pytest.mark.asyncio
async def test_virtual_context_page_out_explicit():
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")
    vcm = VirtualContextManager()

    user_id = "user_123"
    state = PADState(0.1, 0.2, 0.3)

    e1 = MemoryEntry("First Item", 0.5, state, user_id=user_id)
    e2 = MemoryEntry("Second Item", 0.6, state, user_id=user_id)

    await wm.add(e1)
    await wm.add(e2)

    assert len(wm.get_entries(user_id=user_id)) == 2

    # Explicitly page out 1 item
    await vcm.page_out_explicit(wm, em, user_id=user_id, count=1)

    entries = wm.get_entries(user_id=user_id)
    assert len(entries) == 1
    assert entries[0].content == "Second Item"

    # Verify Episodic Memory has the paged out entry
    episodic_events = em.get_all_events(user_id=user_id)
    assert len(episodic_events) == 1
    assert episodic_events[0]["text"] == "First Item"
    assert episodic_events[0]["metadata"]["paged_out_explicitly"] == True

@pytest.mark.asyncio
async def test_virtual_context_page_in_explicit():
    wm = WorkingMemory(limit=5)
    em = EpisodicMemory(persist_directory=":memory:")
    vcm = VirtualContextManager()

    user_id = "user_456"
    em.store_event("Historical facts about Python", metadata={"paged_out": True}, user_id=user_id)

    await vcm.page_in_explicit(wm, em, user_id=user_id, query="Python facts")

    entries = wm.get_entries(user_id=user_id)
    assert len(entries) == 1
    assert "Historical facts about Python" in entries[0].content

@pytest.mark.asyncio
async def test_virtual_context_plugin_assemble():
    vcm = VirtualContextManager()
    user_id = "user_789"
    await vcm.update_core_section(user_id, "persona", "I am a test agent.")

    state = PADState(0, 0, 0)
    e1 = MemoryEntry("Working fact", 0.8, state, user_id=user_id)

    assembled = await vcm.assemble([e1], {"user_id": user_id})

    assert "CORE MEMORY (PERSONA):\nI am a test agent." in assembled
    assert "WORKING MEMORY:\n- Working fact" in assembled

@pytest.mark.asyncio
async def test_virtual_context_plugin_compact():
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "Compressed Summary"
    vcm = VirtualContextManager(llm_client=mock_llm, section_limit=100)

    user_id = "user_abc"
    state = PADState(0.5, 0.5, 0.5)
    e1 = MemoryEntry("Fact 1", 0.5, state, user_id=user_id)
    e2 = MemoryEntry("Fact 2", 0.5, state, user_id=user_id)

    # Threshold 1, so 2 items should trigger compaction
    compacted = await vcm.compact([e1, e2], {"limit": 1, "user_id": user_id})

    assert len(compacted) == 1
    assert compacted[0].content == "Compressed Summary"
    mock_llm.chat_completion.assert_called_once()

@pytest.mark.asyncio
async def test_core_memory_management():
    vcm = VirtualContextManager()
    user_id = "user_999"

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
    user_id = "user_limit"

    long_content = "This is a very long string that will definitely exceed the five word limit."
    await vcm.update_core_section(user_id, "persona", long_content)

    core = vcm.get_core_memory(user_id)
    # Heuristic limit is 5 words
    words = core.persona.split()
    assert len(words) == 5 # limit (last word includes "...")
    assert core.persona.endswith("...")

@pytest.mark.asyncio
async def test_maintain_working_memory_limits_pages_out():
    wm = WorkingMemory(limit=10)
    em = EpisodicMemory(persist_directory=":memory:")
    vcm = VirtualContextManager()

    user_id = "user_maintain"
    state = PADState(0, 0, 0)

    # Each entry has 4 words -> ~5 tokens. 3 entries = ~15 tokens
    e1 = MemoryEntry("word1 word2 word3 word4", 0.5, state, user_id=user_id)
    e2 = MemoryEntry("word5 word6 word7 word8", 0.6, state, user_id=user_id)
    e3 = MemoryEntry("word9 word10 word11 word12", 0.7, state, user_id=user_id)

    await wm.add(e1)
    await wm.add(e2)
    await wm.add(e3)

    assert len(wm.get_entries(user_id=user_id)) == 3

    # Maintain with max_tokens = 10. Current is ~15, so it should page out entries.
    await vcm.maintain_working_memory_limits(wm, em, user_id=user_id, max_tokens=10)

    # It pages out max(1, len(entries)//2) = max(1, 1) = 1 entry.
    entries = wm.get_entries(user_id=user_id)
    assert len(entries) == 2
    assert entries[0].content == "word5 word6 word7 word8"
    assert entries[1].content == "word9 word10 word11 word12"

@pytest.mark.asyncio
async def test_virtual_context_hierarchical_layers():
    # 1. Verify standard instantiation with hierarchical partitioning
    vcm = VirtualContextManager()
    assert isinstance(vcm.working_memory, WorkingMemory)
    assert isinstance(vcm.episodic_memory, EpisodicMemory)
    assert isinstance(vcm.semantic_memory, SemanticMemory)
    assert isinstance(vcm.procedural_memory, ProceduralMemory)

    # 2. Verify wrapper helper methods for Semantic Memory
    vcm.store_fact("Paris is the capital of France", metadata={"category": "geography"}, user_id="user_abc")
    vcm.store_fact("Water boils at 100 degrees Celsius", metadata={"category": "physics"}, user_id="user_abc")

    recalled_facts = vcm.recall_facts("capital", top_k=1, user_id="user_abc")
    assert len(recalled_facts) == 1
    assert "Paris" in recalled_facts[0]

    # 3. Verify wrapper helper methods for Procedural Memory
    vcm.store_procedure("make_coffee", "Boil water, grind beans, brew, serve", metadata={"difficulty": "easy"}, user_id="user_abc")

    recalled_procs = vcm.recall_procedure("make_coffee", top_k=1, user_id="user_abc")
    assert len(recalled_procs) == 1
    assert "coffee" in recalled_procs[0]

    # 4. Verify default fallback behavior (None as argument uses partitioned instances)
    state = PADState(0.1, 0.2, 0.3)
    entry = MemoryEntry("Internal working item", 0.9, state, user_id="user_abc")
    await vcm.working_memory.add(entry)

    # Calling page_out_explicit with None fallback
    await vcm.page_out_explicit(working_memory=None, episodic_memory=None, user_id="user_abc", count=1)

    # Assert working memory is cleared
    assert len(vcm.working_memory.get_entries(user_id="user_abc")) == 0
    # Assert episodic memory has the event
    events = vcm.episodic_memory.get_all_events(user_id="user_abc")
    assert len(events) == 1
    assert events[0]["text"] == "Internal working item"
