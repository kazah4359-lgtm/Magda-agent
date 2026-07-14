import pytest
from unittest.mock import AsyncMock
from typing import Any, List
from magda_agent.memory.context_selective_retrieval_v3 import ContextSelectiveRetrievalV3
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

@pytest.mark.asyncio
async def test_token_length_estimation() -> None:
    """Tests the heuristic token length calculation based on word count."""
    retriever = ContextSelectiveRetrievalV3()
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("word1 word2 word3", 0.5, state)
    e2 = MemoryEntry("word4 word5", 0.6, state)

    # 5 words * 1.3 = 6.5 -> 6 tokens
    assert retriever.get_token_length([e1, e2]) == 6


@pytest.mark.asyncio
async def test_relevance_calculation() -> None:
    """Tests basic keyword overlap-based relevance scoring."""
    retriever = ContextSelectiveRetrievalV3()

    score_full = retriever.calculate_relevance("test query overlap", "test query")
    score_partial = retriever.calculate_relevance("test other", "test query")
    score_none = retriever.calculate_relevance("other text", "test query")

    assert score_full == 1.0
    assert score_partial == 0.5
    assert score_none == 0.0


@pytest.mark.asyncio
async def test_prune_context_under_limit() -> None:
    """Verifies that no pruning occurs if context is under max_tokens."""
    retriever = ContextSelectiveRetrievalV3()
    state = PADState(0, 0, 0)

    entries = [
        MemoryEntry("apple banana cherry", 0.5, state),
        MemoryEntry("date fig grape", 0.6, state)
    ]

    # Total words = 6 -> token length = 7. max_tokens = 10.
    result = await retriever.prune_context(entries, max_tokens=10)
    assert len(result) == 2
    assert result == entries


@pytest.mark.asyncio
async def test_prune_context_with_truncation() -> None:
    """Verifies that non-critical entries are truncated in priority order when LLM is absent."""
    retriever = ContextSelectiveRetrievalV3(importance_threshold=0.8)
    state = PADState(0, 0, 0)

    # max_tokens = 5.
    # Entries:
    e1 = MemoryEntry("banana apple", 0.5, state) # Non-critical
    e2 = MemoryEntry("grape cherry berry", 0.9, state) # Critical (importance >= 0.8)
    e3 = MemoryEntry("kiwi lemon orange", 0.2, state) # Non-critical

    entries = [e1, e2, e3]

    # Total tokens: 8 * 1.3 = 10.
    # Critical e2: tokens = 3 * 1.3 = 3.
    # We want to prune with max_tokens = 6.
    # Critical e2 must be retained.
    result = await retriever.prune_context(entries, max_tokens=6)

    assert len(result) < 3
    assert e2 in result  # Critical entry must be preserved
    assert retriever.get_token_length(result) <= 6


@pytest.mark.asyncio
async def test_prune_context_with_llm_compression() -> None:
    """Tests context compression of non-critical memories using LLM client."""
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "Comp Summary"
    retriever = ContextSelectiveRetrievalV3(llm=mock_llm, importance_threshold=0.8)
    state = PADState(0.1, 0.2, 0.3)

    e1 = MemoryEntry("very long non critical piece one", 0.4, state, tags=["tag1"])
    e2 = MemoryEntry("very long critical piece two", 0.9, state, tags=["tag2"])
    e3 = MemoryEntry("very long non critical piece three", 0.3, state, tags=["tag3"])

    entries = [e1, e2, e3]

    # Total tokens is high. We prune to max_tokens = 12.
    result = await retriever.prune_context(entries, max_tokens=12, query="two")

    # Critical e2 should be present because of its high importance
    assert e2 in result
    # Non-critical items (e1, e3) should be compressed into one MemoryEntry
    assert len(result) == 2

    summary_entry = [e for e in result if e != e2][0]
    assert summary_entry.content == "Comp Summary"
    assert "tag1" in summary_entry.tags
    assert "tag3" in summary_entry.tags


@pytest.mark.asyncio
async def test_context_plugin_compact_lifecycle() -> None:
    """Tests the compact method wrapper to ensure compatibility with ContextPlugin lifecycle."""
    retriever = ContextSelectiveRetrievalV3()
    state = PADState(0, 0, 0)

    entries = [
        MemoryEntry("one two", 0.5, state),
        MemoryEntry("three four", 0.6, state)
    ]

    # Compact to max_tokens=3
    metadata = {"max_tokens": 3}
    result = await retriever.compact(entries, metadata)

    assert len(result) < 2
    assert retriever.get_token_length(result) <= 3
