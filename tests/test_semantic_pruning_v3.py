import pytest
import math
import time
from unittest.mock import AsyncMock, MagicMock

from magda_agent.memory.semantic_pruning_v3 import (
    DecayConfigV3,
    SemanticPruningHookV3,
    SemanticPruningV3,
    _extract_importance,
    _extract_tags,
    _extract_text,
    _extract_timestamp,
)


class DummyEntry:
    def __init__(self, content: str, importance: float = 0.5, timestamp: float = 0.0, tags=None, metadata=None):
        self.content = content
        self.importance = importance
        self.timestamp = timestamp
        self.tags = tags or []
        self.metadata = metadata or {}


def test_decay_config_defaults_and_custom():
    default_cfg = DecayConfigV3()
    assert default_cfg.decay_rate == 0.05
    assert default_cfg.half_life_seconds == 3600.0
    assert default_cfg.min_utility_threshold == 0.35
    assert default_cfg.decay_mode == "exponential"
    assert default_cfg.weight_recency == 0.4
    assert default_cfg.weight_importance == 0.3
    assert default_cfg.weight_relevance == 0.3

    custom_cfg = DecayConfigV3(
        half_life_seconds=1800.0,
        min_utility_threshold=0.5,
        decay_mode="linear",
        weight_recency=0.5,
        weight_importance=0.25,
        weight_relevance=0.25,
    )
    assert custom_cfg.decay_mode == "linear"
    assert custom_cfg.half_life_seconds == 1800.0


def test_helper_extractions():
    # Text extraction
    assert _extract_text("simple text") == "simple text"
    assert _extract_text({"content": "dict content"}) == "dict content"
    assert _extract_text({"text": "dict text"}) == "dict text"

    obj = DummyEntry("object content")
    assert _extract_text(obj) == "object content"

    # Timestamp extraction
    t0 = 1000.0
    assert _extract_timestamp({"timestamp": t0}) == t0
    assert _extract_timestamp({"created_at": t0 + 10}) == t0 + 10
    assert _extract_timestamp({"metadata": {"timestamp": t0 + 20}}) == t0 + 20
    assert _extract_timestamp(DummyEntry("t", timestamp=500.0)) == 500.0
    assert _extract_timestamp("no time", default_time=123.0) == 123.0

    # Importance extraction
    assert _extract_importance({"importance": 0.9}) == 0.9
    assert _extract_importance({"metadata": {"importance": 0.85}}) == 0.85
    assert _extract_importance(DummyEntry("imp", importance=0.75)) == 0.75
    assert _extract_importance("no imp", default=0.5) == 0.5

    # Tags extraction
    d_tags = {"tags": ["AI", "Memory"]}
    assert _extract_tags(d_tags) == ["ai", "memory"]

    obj_tags = DummyEntry("tag text #debug #alert", tags=["system"])
    tags_res = _extract_tags(obj_tags)
    assert "system" in tags_res
    assert "debug" in tags_res
    assert "alert" in tags_res


def test_recency_score_exponential_decay():
    cfg = DecayConfigV3(half_life_seconds=3600.0, decay_mode="exponential")
    hook = SemanticPruningHookV3(config=cfg)

    t0 = 10000.0
    # Exactly now
    assert hook.compute_recency_score(t0, current_time=t0) == pytest.approx(1.0)

    # After 1 half-life
    assert hook.compute_recency_score(t0, current_time=t0 + 3600.0) == pytest.approx(0.5, rel=1e-3)

    # After 2 half-lives
    assert hook.compute_recency_score(t0, current_time=t0 + 7200.0) == pytest.approx(0.25, rel=1e-3)


def test_recency_score_linear_decay():
    # Horizon is 2 * half_life = 7200
    cfg = DecayConfigV3(half_life_seconds=3600.0, decay_mode="linear")
    hook = SemanticPruningHookV3(config=cfg)

    t0 = 10000.0
    assert hook.compute_recency_score(t0, current_time=t0) == 1.0
    assert hook.compute_recency_score(t0, current_time=t0 + 3600.0) == pytest.approx(0.5, rel=1e-3)
    assert hook.compute_recency_score(t0, current_time=t0 + 7200.0) == pytest.approx(0.0, rel=1e-3)
    assert hook.compute_recency_score(t0, current_time=t0 + 10000.0) == 0.0


def test_semantic_relevance_heuristic():
    hook = SemanticPruningHookV3()
    # No query
    assert hook.compute_semantic_relevance("some text", query=None) == 1.0
    assert hook.compute_semantic_relevance("some text", query="") == 1.0

    # Overlap
    rel_full = hook.compute_semantic_relevance("Fix bug in database connection", query="database connection")
    assert rel_full == 1.0

    rel_partial = hook.compute_semantic_relevance("Fix bug in database connection", query="database redis memory")
    assert 0.0 < rel_partial < 1.0

    rel_none = hook.compute_semantic_relevance("Fix bug in database connection", query="frontend react css")
    assert rel_none == 0.0


@pytest.mark.asyncio
async def test_semantic_relevance_llm():
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value="0.85")

    hook = SemanticPruningHookV3(llm_client=mock_llm)
    score = await hook.compute_semantic_relevance_llm("memory text", query="search topic")
    assert score == pytest.approx(0.85)

    # Fallback when LLM raises error
    mock_llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    fallback_score = await hook.compute_semantic_relevance_llm("search topic in text", query="search topic")
    assert fallback_score == 1.0


def test_calculate_utility():
    cfg = DecayConfigV3(
        weight_recency=0.4,
        weight_importance=0.3,
        weight_relevance=0.3,
        half_life_seconds=3600.0,
    )
    hook = SemanticPruningHookV3(config=cfg)

    now = 50000.0
    entry = DummyEntry(
        content="Important database fix",
        importance=1.0,
        timestamp=now,
    )
    utility, metrics = hook.calculate_utility(entry, query="database fix", current_time=now)
    assert utility == pytest.approx(1.0)
    assert metrics["recency"] == pytest.approx(1.0)
    assert metrics["importance"] == pytest.approx(1.0)
    assert metrics["relevance"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_prune_context_dynamic_decay():
    cfg = DecayConfigV3(
        half_life_seconds=3600.0,
        min_utility_threshold=0.45,
        decay_mode="exponential",
    )
    hook = SemanticPruningHookV3(config=cfg)

    now = 100000.0
    # Fresh high-importance entry
    fresh_item = DummyEntry("Recent important discussion", importance=0.9, timestamp=now)
    # Old low-importance entry (decayed)
    decayed_item = DummyEntry("Old trivial log entry", importance=0.1, timestamp=now - 72000.0)

    items = [fresh_item, decayed_item]
    retained = await hook.prune_context(items, current_time=now)

    assert len(retained) == 1
    assert retained[0] == fresh_item
    assert len(hook.pruning_history) == 1
    assert hook.pruning_history[0]["item"] == "Old trivial log entry"


@pytest.mark.asyncio
async def test_offload_to_episodic_memory():
    mock_store = MagicMock()
    hook = SemanticPruningHookV3(
        config=DecayConfigV3(min_utility_threshold=0.5),
        episodic_memory=mock_store,
    )

    now = 100000.0
    decayed_item = DummyEntry("Old irrelevant item #prune", importance=0.1, timestamp=now - 50000.0, tags=["prune"])
    retained = await hook.prune_context([decayed_item], current_time=now, metadata={"user_id": 42})

    assert len(retained) == 0
    assert mock_store.store_event.call_count == 1
    mock_store.store_event.assert_called_with(
        "Old irrelevant item #prune",
        metadata={"tags": ["prune"], "pruned_by": "SemanticPruningHookV3", "pruned": True},
        user_id=42,
    )


@pytest.mark.asyncio
async def test_max_context_items_enforcement():
    hook = SemanticPruningHookV3(
        config=DecayConfigV3(min_utility_threshold=0.1),
        max_context_items=2,
    )

    now = 100000.0
    i1 = DummyEntry("Item 1", importance=0.9, timestamp=now)
    i2 = DummyEntry("Item 2", importance=0.8, timestamp=now)
    i3 = DummyEntry("Item 3", importance=0.3, timestamp=now)

    retained = await hook.prune_context([i1, i2, i3], current_time=now)
    assert len(retained) == 2
    assert i1 in retained
    assert i2 in retained
    assert i3 not in retained


@pytest.mark.asyncio
async def test_context_plugin_lifecycle_methods():
    hook = SemanticPruningV3()

    # Bootstrap
    await hook.bootstrap({
        "half_life_seconds": 1800.0,
        "min_utility_threshold": 0.4,
        "decay_mode": "linear",
    })
    assert hook.config.half_life_seconds == 1800.0
    assert hook.config.min_utility_threshold == 0.4
    assert hook.config.decay_mode == "linear"

    # Ingest
    ingested = await hook.ingest("Content test", {"tags": ["test"]})
    assert "[PruningTracked:test]" in ingested

    # Assemble
    items = [DummyEntry("Item A"), DummyEntry("Item B")]
    assembled = await hook.assemble(items, {})
    assert assembled == "Item A\nItem B"

    # Compact via ContextPlugin
    compacted = await hook.compact(items, {"min_utility": 0.0})
    assert len(compacted) == 2

    # Sync hooks
    assert hook.before_retrieval("q", 1) == "q"
    assert hook.after_retrieval(["c"], "q", 1) == ["c"]
    assert hook.before_write("w", 1) == "w"
    hook.after_write("w", 1)
    hook.on_context_update("ctx", 1)
