import pytest
from magda_agent.memory.virtual_memory import VirtualMemoryPlugin
from magda_agent.memory.working import MemoryEntry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_virtual_memory_bootstrap() -> None:
    """Test bootstrapping VirtualMemoryPlugin updates its configuration."""
    plugin = VirtualMemoryPlugin(token_limit=1000)
    config = {"token_limit": 500}
    await plugin.bootstrap(config)
    assert plugin.token_limit == 500

@pytest.mark.asyncio
async def test_virtual_memory_assemble() -> None:
    """Test assemble constructs correct context prompt."""
    plugin = VirtualMemoryPlugin()
    entry1 = MemoryEntry("Hello world", 0.5, PADState(0.0, 0.0, 0.0))
    entry2 = MemoryEntry("Context test", 0.8, PADState(0.0, 0.0, 0.0))
    result = await plugin.assemble([entry1, entry2], {})
    assert "VIRTUAL CONTEXT:" in result
    assert "- Hello world" in result
    assert "- Context test" in result

@pytest.mark.asyncio
async def test_virtual_memory_compact_by_count() -> None:
    """Test compaction based on item limit."""
    episodic_mem = EpisodicMemory(persist_directory=":memory:")
    plugin = VirtualMemoryPlugin(episodic_memory=episodic_mem)

    # Use a highly unique user_id to prevent cross-test contamination in in-memory ChromaDB
    user_id = 998765432

    entries = [
        MemoryEntry(f"Item {i}", 0.5, PADState(0.1, 0.2, 0.3), user_id=user_id) for i in range(15)
    ]

    result = await plugin.compact(entries, {"limit": 10, "user_id": user_id})
    assert len(result) == 10
    # First 5 items (0 to 4) should be swapped out to episodic memory
    assert result[0].content == "Item 5"

    events = episodic_mem.get_all_events(user_id=user_id)
    assert len(events) == 5
    assert events[0]["text"] == "Item 0"
    assert events[0]["metadata"]["swapped_from_virtual"] is True
    assert events[0]["metadata"]["pad_p"] == 0.1

@pytest.mark.asyncio
async def test_virtual_memory_compact_by_tokens() -> None:
    """Test compaction based on token limits."""
    episodic_mem = EpisodicMemory(persist_directory=":memory:")
    # Small token limit (19) to trigger compaction and force dropping below 20 tokens
    plugin = VirtualMemoryPlugin(episodic_memory=episodic_mem, token_limit=19)

    user_id = 998765433

    # Each entry has 4 words -> ~5.2 tokens
    entries = [
        MemoryEntry("one two three four", 0.5, PADState(0.0, 0.0, 0.0), user_id=user_id),
        MemoryEntry("five six seven eight", 0.5, PADState(0.0, 0.0, 0.0), user_id=user_id),
        MemoryEntry("nine ten eleven twelve", 0.5, PADState(0.0, 0.0, 0.0), user_id=user_id),
        MemoryEntry("thirteen fourteen fifteen sixteen", 0.5, PADState(0.0, 0.0, 0.0), user_id=user_id),
        MemoryEntry("seventeen eighteen nineteen twenty", 0.5, PADState(0.0, 0.0, 0.0), user_id=user_id),
    ]

    # Item limit is high (10), but token limit should force swap outs
    result = await plugin.compact(entries, {"limit": 10, "user_id": user_id})
    # Total tokens is (5 * 4 * 1.3) = 26.
    # To drop below 19 tokens, we must swap out at least 2 entries (each is ~5 tokens, leaving 3 entries with 15 tokens).
    assert len(result) <= 3

    events = episodic_mem.get_all_events(user_id=user_id)
    assert len(events) >= 2
    assert events[0]["metadata"]["swapped_from_virtual"] is True

@pytest.mark.asyncio
async def test_virtual_memory_swap_in() -> None:
    """Test swapping in relevant memories."""
    episodic_mem = EpisodicMemory(persist_directory=":memory:")
    plugin = VirtualMemoryPlugin(episodic_memory=episodic_mem)

    user_id = 998765434

    # Seed episodic memory
    episodic_mem.store_event("Deep Learning is awesome", metadata={"swapped_from_virtual": True}, user_id=user_id)
    episodic_mem.store_event("Context length matters", metadata={"swapped_from_virtual": True}, user_id=user_id)

    current_context = [MemoryEntry("Regular dialogue", 0.5, PADState(0.0, 0.0, 0.0), user_id=user_id)]
    updated_context = await plugin.swap_in(current_context, "Deep Learning", top_k=2, user_id=user_id)

    # "Deep Learning is awesome" is more relevant and should be swapped in
    contents = [e.content for e in updated_context]
    assert len(updated_context) > 1
    assert "Deep Learning is awesome" in contents
