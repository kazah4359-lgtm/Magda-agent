import pytest
import time
from unittest.mock import MagicMock, AsyncMock
from magda_agent.memory.semantic_pruning_v3 import (
    DecayConfigV3,
    SemanticPruningHookV3,
    SemanticPruningV3,
)
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState


def test_decay_factor_calculation() -> None:
    """Verifies exponential and linear decay factor calculations."""
    config_exp = DecayConfigV3(half_life_seconds=3600.0, decay_mode="exponential")
    hook_exp = SemanticPruningHookV3(config=config_exp)

    # At t = 0, decay factor should be 1.0
    assert hook_exp.calculate_decay_factor(0.0) == 1.0
    # At t = 3600 (one half-life), decay factor should be 0.5
    assert pytest.approx(hook_exp.calculate_decay_factor(3600.0), 0.001) == 0.5
    # At t = 7200 (two half-lives), decay factor should be 0.25
    assert pytest.approx(hook_exp.calculate_decay_factor(7200.0), 0.001) == 0.25

    # Linear decay
    config_lin = DecayConfigV3(decay_mode="linear", linear_decay_rate=0.0001)
    hook_lin = SemanticPruningHookV3(config=config_lin)
    assert hook_lin.calculate_decay_factor(0.0) == 1.0
    # At t = 5000, 1.0 - (0.0001 * 5000) = 0.5
    assert pytest.approx(hook_lin.calculate_decay_factor(5000.0), 0.001) == 0.5
    # At t = 10000, 1.0 - 1.0 = 0.0
    assert hook_lin.calculate_decay_factor(10000.0) == 0.0


def test_semantic_relevance_calculation() -> None:
    """Verifies word overlap scoring."""
    hook = SemanticPruningHookV3()
    # No query defaults to 0.5
    assert hook.calculate_semantic_relevance("test content", None) == 0.5
    assert hook.calculate_semantic_relevance("test content", "") == 0.5

    # Full overlap
    assert hook.calculate_semantic_relevance("python async agent memory", "python memory") == 1.0
    # Partial overlap
    assert hook.calculate_semantic_relevance("python agent", "python memory") == 0.5
    # No overlap
    assert hook.calculate_semantic_relevance("java spring boot", "python memory") == 0.0


@pytest.mark.asyncio
async def test_prune_context_decayed_entries() -> None:
    """Verifies that entries with decayed utility below min_utility_threshold are pruned."""
    config = DecayConfigV3(
        half_life_seconds=100.0,
        min_utility_threshold=0.3,
        importance_weight=0.5,
        relevance_weight=0.3,
        recency_weight=0.2,
    )
    hook = SemanticPruningHookV3(config=config)
    pad = PADState(0, 0, 0)

    now = 10000.0
    # Old entry from 1000 seconds ago (10 half-lives, decayed to near 0)
    old_entry = MemoryEntry("unrelated old discussion", importance=0.4, emotional_state=pad)
    old_entry.timestamp = now - 1000.0

    # Fresh entry just created
    fresh_entry = MemoryEntry("active project tasks", importance=0.6, emotional_state=pad)
    fresh_entry.timestamp = now

    entries = [old_entry, fresh_entry]
    pruned = await hook.prune_context(
        entries=entries,
        current_time=now,
        query="active project",
    )

    # old_entry should be pruned, fresh_entry should be retained
    assert fresh_entry in pruned
    assert old_entry not in pruned
    assert len(pruned) == 1
    assert len(hook.pruned_history) == 1


@pytest.mark.asyncio
async def test_preserve_critical_entries() -> None:
    """Critical entries (importance >= 0.8) are preserved even when older."""
    config = DecayConfigV3(
        half_life_seconds=50.0,
        min_utility_threshold=0.5,
    )
    hook = SemanticPruningHookV3(config=config)
    pad = PADState(0, 0, 0)

    now = 20000.0
    # Critical entry with high importance
    critical_entry = MemoryEntry("User security password rule", importance=0.9, emotional_state=pad)
    critical_entry.timestamp = now - 500.0  # Decayed significantly in time

    # Non-critical entry
    non_critical = MemoryEntry("casual conversation chit chat", importance=0.3, emotional_state=pad)
    non_critical.timestamp = now - 500.0

    entries = [critical_entry, non_critical]
    pruned = await hook.prune_context(entries=entries, current_time=now)

    assert critical_entry in pruned
    assert non_critical not in pruned


@pytest.mark.asyncio
async def test_token_limit_pruning() -> None:
    """Verifies that entries with lowest utility are evicted when token limit is exceeded."""
    config = DecayConfigV3(min_utility_threshold=0.1)
    hook = SemanticPruningHookV3(config=config)
    pad = PADState(0, 0, 0)

    now = time.time()
    e1 = MemoryEntry("one two three four five", importance=0.4, emotional_state=pad)
    e1.timestamp = now - 200.0

    e2 = MemoryEntry("six seven eight nine ten", importance=0.7, emotional_state=pad)
    e2.timestamp = now

    entries = [e1, e2]
    # e1 and e2 are each 6 tokens (5 words * 1.3 = 6.5 -> 6). Total = 12 tokens.
    # Prune with max_tokens = 8 so that only one entry fits.
    pruned = await hook.prune_context(entries=entries, max_tokens=8, current_time=now)

    assert len(pruned) == 1
    assert pruned[0] == e2  # e2 has higher utility score


@pytest.mark.asyncio
async def test_offload_to_episodic_memory() -> None:
    """Verifies that pruned entries are offloaded to episodic memory."""
    mock_episodic = MagicMock()
    hook = SemanticPruningHookV3(
        config=DecayConfigV3(min_utility_threshold=0.5),
        episodic_memory=mock_episodic,
    )
    pad = PADState(0, 0, 0)

    now = 1000.0
    old_entry = MemoryEntry("evicted note", importance=0.2, emotional_state=pad, user_id=42)
    old_entry.timestamp = now - 5000.0

    await hook.prune_context([old_entry], current_time=now, user_id=42)

    mock_episodic.store_event.assert_called_once()
    call_args = mock_episodic.store_event.call_args[1]
    assert call_args["text"] == "evicted note"
    assert call_args["user_id"] == 42
    assert call_args["metadata"]["pruned_by"] == "SemanticPruningHookV3"


@pytest.mark.asyncio
async def test_context_plugin_lifecycle() -> None:
    """Verifies ContextPlugin methods (bootstrap, compact, assemble, ingest, etc.)."""
    hook = SemanticPruningV3()
    pad = PADState(0, 0, 0)

    # bootstrap
    await hook.bootstrap({
        "half_life_seconds": 1800.0,
        "decay_mode": "linear",
        "min_utility_threshold": 0.4,
        "default_max_tokens": 1500,
    })
    assert hook.config.half_life_seconds == 1800.0
    assert hook.config.decay_mode == "linear"
    assert hook.config.min_utility_threshold == 0.4
    assert hook.default_max_tokens == 1500

    # ingest & assemble
    assert await hook.ingest("raw text", {}) == "raw text"
    assembled = await hook.assemble([MemoryEntry("item1", 0.5, pad), MemoryEntry("item2", 0.5, pad)], {})
    assert "- item1" in assembled
    assert "- item2" in assembled

    # compact hook
    now = time.time()
    e_decayed = MemoryEntry("decayed event", importance=0.1, emotional_state=pad)
    e_decayed.timestamp = now - 100000.0

    e_active = MemoryEntry("active workflow event", importance=0.8, emotional_state=pad)
    e_active.timestamp = now

    compacted = await hook.compact([e_decayed, e_active], {"max_tokens": 2000, "current_time": now})
    assert e_active in compacted
    assert e_decayed not in compacted

    # Synchronous lifecycle hooks
    assert hook.before_retrieval("search query", 1) == "search query"
    assert hook.after_retrieval(["item"], "search query", 1) == ["item"]
    assert hook.before_write("entry", 1) == "entry"
    hook.after_write("entry", 1)
    hook.on_context_update("ctx", 1)
    assert await hook.pre_process("content", {}) == "content"
    assert await hook.post_process("response", {}) == "response"
