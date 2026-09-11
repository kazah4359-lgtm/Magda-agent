import pytest
import asyncio
from magda_agent.memory.semantic_pager_v2 import (
    SemanticClusterPagerV2,
    default_cosine_similarity,
    vector_cosine_similarity,
)
from magda_agent.memory.context_engine import ContextEngine


def mock_embedding_function(text: str):
    """
    Mock local embedding function that maps specific themes to orthogonal vectors.
    """
    text_lower = text.lower()
    if "python" in text_lower or "code" in text_lower or "script" in text_lower:
        return [1.0, 0.0, 0.0]
    elif "cooking" in text_lower or "recipe" in text_lower or "food" in text_lower:
        return [0.0, 1.0, 0.0]
    elif "space" in text_lower or "astronomy" in text_lower or "planet" in text_lower:
        return [0.0, 0.0, 1.0]
    else:
        return [0.5, 0.5, 0.5]


@pytest.mark.asyncio
async def test_cluster_memories_mock_embeddings():
    pager = SemanticClusterPagerV2(
        similarity_threshold=0.8,
        embedding_fn=mock_embedding_function,
    )

    memories = [
        "Python coding standard script",
        "Recipe for cooking delicious food",
        "Python async code function",
        "Astronomy and space planet exploration",
        "Cooking food recipe tutorial",
    ]

    clusters = pager.cluster_memories(memories)

    assert len(clusters) == 3

    python_cluster = [c for c in clusters if any("Python" in item for item in c)]
    assert len(python_cluster) == 1
    assert len(python_cluster[0]) == 2

    cooking_cluster = [c for c in clusters if any("Cooking" in item or "Recipe" in item for item in c)]
    assert len(cooking_cluster) == 1
    assert len(cooking_cluster[0]) == 2


@pytest.mark.asyncio
async def test_evict_redundant_clusters_concurrently():
    pager = SemanticClusterPagerV2(
        similarity_threshold=0.8,
        max_items=3,
        mode="evict",
        embedding_fn=mock_embedding_function,
    )

    memories = [
        "Python coding standard script 1",
        "Python coding standard script 2",
        "Python coding standard script 3",
        "Recipe for cooking food",
        "Space planet exploration",
    ]

    compacted = await pager.page_and_compact(memories)

    assert len(compacted) == 3
    assert "Python coding standard script 3" in compacted


@pytest.mark.asyncio
async def test_summarize_redundant_clusters_concurrently():
    async def mock_summarize(cluster):
        return f"Summary of {len(cluster)} python items"

    pager = SemanticClusterPagerV2(
        similarity_threshold=0.8,
        max_items=3,
        mode="summarize",
        embedding_fn=mock_embedding_function,
        summarize_fn=mock_summarize,
    )

    memories = [
        "Python script 1",
        "Python script 2",
        "Python script 3",
        "Cooking food recipe",
        "Space planet search",
    ]

    compacted = await pager.page_and_compact(memories)

    assert len(compacted) == 3
    assert "Summary of 3 python items" in compacted


@pytest.mark.asyncio
async def test_context_engine_integration():
    pager = SemanticClusterPagerV2(
        similarity_threshold=0.8,
        max_items=2,
        mode="evict",
        embedding_fn=mock_embedding_function,
    )

    engine = ContextEngine(plugins=[pager])

    items = [
        {"content": "Python code snippet A"},
        {"content": "Python code snippet B"},
        {"content": "Space research mission"},
    ]

    compacted = await engine.compact(items, {"max_items": 2})

    assert len(compacted) == 2
    contents = [x["content"] if isinstance(x, dict) else str(x) for x in compacted]
    assert "Python code snippet B" in contents
    assert "Space research mission" in contents


@pytest.mark.asyncio
async def test_edge_cases():
    pager = SemanticClusterPagerV2(
        similarity_threshold=0.8,
        max_items=5,
        max_tokens=1000,
    )

    # Empty
    assert await pager.page_and_compact([]) == []

    # Single item
    single = ["Only one memory"]
    assert await pager.page_and_compact(single) == ["Only one memory"]

    # Threshold not exceeded
    items = ["Memory 1", "Memory 2"]
    assert await pager.page_and_compact(items) == items
