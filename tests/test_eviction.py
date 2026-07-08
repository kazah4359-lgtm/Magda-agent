import pytest
import asyncio
from unittest.mock import AsyncMock
from magda_agent.memory.eviction import EvictionPolicy
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

class MockMemoryEntry:
    def __init__(self, content: str, importance: float):
        self.content = content
        self.importance = importance

@pytest.mark.asyncio
async def test_eviction_policy_no_eviction_needed() -> None:
    """Test that EvictionPolicy does nothing if items are within limit."""
    policy = EvictionPolicy()
    items = [
        MockMemoryEntry("msg1", 0.8),
        MockMemoryEntry("msg2", 0.5)
    ]

    result = await policy.compact(items, {"limit": 5})
    assert len(result) == 2
    assert result == items

@pytest.mark.asyncio
async def test_eviction_policy_drops_lowest_importance() -> None:
    """Test that EvictionPolicy drops the lowest importance items when limit is reached."""
    policy = EvictionPolicy()

    items = [
        MockMemoryEntry("oldest_high", 0.9),
        MockMemoryEntry("oldest_low", 0.1),
        MockMemoryEntry("middle_mid", 0.5),
        MockMemoryEntry("newest_high", 0.8),
        MockMemoryEntry("newest_lowest", 0.05)
    ]

    # Limit is 3. We have 5 items. The 2 with lowest importance should be dropped.
    # Those are "oldest_low" (0.1) and "newest_lowest" (0.05).
    # Remaining should be "oldest_high", "middle_mid", "newest_high" in original order.

    result = await policy.compact(items, {"limit": 3})

    assert len(result) == 3
    assert result[0].content == "oldest_high"
    assert result[1].content == "middle_mid"
    assert result[2].content == "newest_high"

@pytest.mark.asyncio
async def test_eviction_policy_tie_breaking() -> None:
    """Test that tie breaking on importance keeps newer items and drops older ones."""
    policy = EvictionPolicy()

    items = [
        MockMemoryEntry("oldest", 0.5),
        MockMemoryEntry("older", 0.5),
        MockMemoryEntry("newer", 0.5),
        MockMemoryEntry("newest", 0.5)
    ]

    # Limit is 2. We have 4 items with equal importance.
    # Should drop the oldest 2 ("oldest", "older").
    # Keep "newer", "newest".

    result = await policy.compact(items, {"limit": 2})

    assert len(result) == 2
    assert result[0].content == "newer"
    assert result[1].content == "newest"

@pytest.mark.asyncio
async def test_eviction_policy_with_real_memory_entry() -> None:
    """Test using actual MemoryEntry objects."""
    policy = EvictionPolicy()
    pad = PADState(0, 0, 0)

    items = [
        MemoryEntry("A", importance=0.9, emotional_state=pad),
        MemoryEntry("B", importance=0.2, emotional_state=pad),
        MemoryEntry("C", importance=0.8, emotional_state=pad)
    ]

    result = await policy.compact(items, {"limit": 2})

    assert len(result) == 2
    assert result[0].content == "A"
    assert result[1].content == "C"
