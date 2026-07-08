"""Tests for TaintedWorkingMemory integration."""
import pytest
from magda_agent.emotions.engine import PADState
from magda_agent.memory.working import MemoryEntry
from magda_agent.safety.taint_tracking_v2 import TaintTrackerV2
from magda_agent.safety.taint_context_v6 import TaintedWorkingMemory

@pytest.mark.asyncio
async def test_tainted_working_memory():
    """Test that taint tags propagate to working memory entries."""
    tracker = TaintTrackerV2()
    memory = TaintedWorkingMemory(tracker=tracker)

    # Test tainted entry
    tainted_str = tracker.taint("malicious_payload", "user_input")
    state = PADState()
    entry1 = MemoryEntry(content=tainted_str, importance=0.8, emotional_state=state)

    await memory.add(entry1)

    assert "tainted" in entry1.tags
    assert len(memory.get_entries()) == 1

    # Test untainted entry
    untainted_str = "normal_payload"
    entry2 = MemoryEntry(content=untainted_str, importance=0.5, emotional_state=state)

    await memory.add(entry2)

    assert "tainted" not in entry2.tags
    assert len(memory.get_entries()) == 2
