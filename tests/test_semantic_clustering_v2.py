import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.memory.semantic_clustering_v2 import (
    SemanticClusteringV2,
    MemoryClusterEntry,
    _extract_tags,
    _extract_text,
    _estimate_tokens,
)


class DummyMemoryEntry:
    def __init__(self, content: str, tags=None, metadata=None):
        self.content = content
        self.tags = tags or []
        self.metadata = metadata or {}


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="LLM generated cluster summary for debugging")
    return llm


@pytest.fixture
def plugin(mock_llm):
    return SemanticClusteringV2(llm=mock_llm, max_tokens=100)


@pytest.mark.asyncio
async def test_plugin_bootstrap(plugin):
    config = {
        "similarity_threshold": 0.8,
        "max_tokens": 50,
        "max_items": 5,
        "mode": "evict",
    }
    await plugin.bootstrap(config)
    assert plugin.similarity_threshold == 0.8
    assert plugin.max_tokens == 50
    assert plugin.max_items == 5
    assert plugin.mode == "evict"


def test_extract_tags():
    # From dict with tags list
    d1 = {"content": "data", "tags": ["Python", "Backend"]}
    assert _extract_tags(d1) == ["python", "backend"]

    # From object with tags attribute
    obj1 = DummyMemoryEntry("some text #debug #fix", tags=["Authentication"])
    assert "authentication" in _extract_tags(obj1)
    assert "debug" in _extract_tags(obj1)
    assert "fix" in _extract_tags(obj1)

    # Fallback to general
    obj_no_tags = DummyMemoryEntry("plain memory text")
    assert _extract_tags(obj_no_tags) == ["general"]


def test_extract_text_and_estimate_tokens():
    text_str = "This is a simple test text"
    tokens_str = _estimate_tokens(text_str)
    assert tokens_str > 0

    entry_obj = DummyMemoryEntry("First word second word third word fourth word")
    tokens_obj = _estimate_tokens(entry_obj)
    assert tokens_obj == int(8 * 1.3)

    items_list = [entry_obj, "Another text entry"]
    tokens_list = _estimate_tokens(items_list)
    assert tokens_list > tokens_obj


def test_cluster_by_tags(plugin):
    m1 = DummyMemoryEntry("Working on python bug #python", tags=["python"])
    m2 = DummyMemoryEntry("Refactoring python script", tags=["python"])
    m3 = DummyMemoryEntry("Configuring docker container", tags=["docker"])

    clusters = plugin.cluster_by_tags([m1, m2, m3])
    assert "python" in clusters
    assert "docker" in clusters
    assert len(clusters["python"]) == 2
    assert len(clusters["docker"]) == 1


@pytest.mark.asyncio
async def test_cluster_and_compress_token_reduction(plugin):
    # Create multiple large episodic memories under same tag
    m1 = DummyMemoryEntry(
        "Detailed memory log about database migration issue with postgres table columns and locks",
        tags=["database"],
    )
    m2 = DummyMemoryEntry(
        "Detailed memory log about database connection pool timeout issues during high load",
        tags=["database"],
    )
    m3 = DummyMemoryEntry(
        "Detailed memory log about database indexing strategy optimization for slow queries",
        tags=["database"],
    )

    memories = [m1, m2, m3]
    initial_tokens = plugin.estimate_tokens(memories)

    compressed = await plugin.cluster_and_compress(memories, max_tokens=100)
    final_tokens = plugin.estimate_tokens(compressed)

    assert len(compressed) < len(memories)
    assert final_tokens < initial_tokens
    assert isinstance(compressed[0], MemoryClusterEntry)
    assert "database" in compressed[0].tags


@pytest.mark.asyncio
async def test_mock_episodic_store_offloading():
    mock_store = MagicMock()
    plugin = SemanticClusteringV2(episodic_memory=mock_store, max_tokens=100)

    m1 = DummyMemoryEntry("Event 1 log #auth", tags=["auth"])
    m2 = DummyMemoryEntry("Event 2 log #auth", tags=["auth"])

    res = await plugin.compact([m1, m2], {"user_id": 123})

    assert len(res) == 1
    assert mock_store.store_event.call_count >= 2
    mock_store.store_event.assert_any_call(
        "Event 1 log #auth",
        metadata={"tags": ["auth"], "clustered": True},
        user_id=123,
    )


@pytest.mark.asyncio
async def test_mock_store_add_memory_and_append():
    mock_store_add = MagicMock(spec=["add_memory"])
    plugin_add = SemanticClusteringV2(episodic_memory=mock_store_add)

    m1 = DummyMemoryEntry("Log A", tags=["test"])
    m2 = DummyMemoryEntry("Log B", tags=["test"])

    await plugin_add.compact([m1, m2], {})
    assert mock_store_add.add_memory.call_count >= 2

    mock_list = []
    plugin_list = SemanticClusteringV2(episodic_memory=mock_list)
    await plugin_list.compact([m1, m2], {})
    assert len(mock_list) >= 2


@pytest.mark.asyncio
async def test_evict_mode():
    plugin_evict = SemanticClusteringV2(mode="evict")
    m1 = DummyMemoryEntry("Older log entry #logging", tags=["logging"])
    m2 = DummyMemoryEntry("Newer log entry #logging", tags=["logging"])

    res = await plugin_evict.cluster_and_compress([m1, m2])
    assert len(res) == 1
    assert res[0] == m2


@pytest.mark.asyncio
async def test_context_plugin_lifecycle_methods(plugin):
    # Ingest
    ingested = await plugin.ingest("Content body", {"tags": ["bug"]})
    assert ingested == "[Tags:bug] Content body"

    # Assemble
    items = [DummyMemoryEntry("Item text", tags=["tag1"])]
    assembled = await plugin.assemble(items, {})
    assert "[tag1] Item text" in assembled

    # Sync hooks
    assert plugin.before_retrieval("query", 1) == "query"
    assert plugin.after_retrieval(["ctx"], "query", 1) == ["ctx"]
    assert plugin.before_write("write_ctx", 1) == "write_ctx"
    plugin.after_write("write_ctx", 1)
    plugin.on_context_update("update", 1)

    # Pre/Post process
    assert await plugin.pre_process("raw", {}) == "raw"
    assert await plugin.post_process("resp", {}) == "resp"
