import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.context_engine_local import LocalFirstContextEngine
from magda_agent.memory.working import MemoryEntry

@pytest.mark.asyncio
async def test_local_first_context_engine_fallback_to_local_llm() -> None:
    """Tests that LocalFirstContextEngine falls back to the local LLM if the primary LLM fails."""

    primary_llm = AsyncMock()
    primary_llm.chat_completion.side_effect = Exception("Primary API down")

    local_llm = AsyncMock()
    local_llm.chat_completion.return_value = "local summary"

    engine = LocalFirstContextEngine(llm=primary_llm, local_llm=local_llm)

    entry1 = MagicMock(spec=MemoryEntry)
    entry1.content = "content1"
    entry1.importance = 0.5
    entry1.tags = ["tag1"]

    entry2 = MagicMock(spec=MemoryEntry)
    entry2.content = "content2"
    entry2.importance = 0.5
    entry2.tags = ["tag2"]

    items = [entry1, entry2]
    # Limit is 1, so compaction will trigger
    compacted = await engine.compact(items, {"limit": 1})

    assert primary_llm.chat_completion.called
    assert local_llm.chat_completion.called
    assert len(compacted) == 1
    assert compacted[0].content == "local summary"
    assert set(compacted[0].tags) == {"tag1", "tag2"}

@pytest.mark.asyncio
async def test_local_first_context_engine_fallback_to_truncation() -> None:
    """Tests that LocalFirstContextEngine truncates if both LLMs fail."""

    primary_llm = AsyncMock()
    primary_llm.chat_completion.side_effect = Exception("Primary API down")

    local_llm = AsyncMock()
    local_llm.chat_completion.side_effect = Exception("Local API down")

    engine = LocalFirstContextEngine(llm=primary_llm, local_llm=local_llm)

    entry1 = MagicMock(spec=MemoryEntry)
    entry1.content = "content1"

    entry2 = MagicMock(spec=MemoryEntry)
    entry2.content = "content2"

    items = [entry1, entry2]
    # Limit is 1, so compaction will trigger
    compacted = await engine.compact(items, {"limit": 1})

    assert primary_llm.chat_completion.called
    assert local_llm.chat_completion.called

    # Should drop the oldest item (entry1)
    assert len(compacted) == 1
    assert compacted[0] == entry2

@pytest.mark.asyncio
async def test_local_first_context_engine_primary_llm_succeeds() -> None:
    """Tests that LocalFirstContextEngine uses the primary LLM if it succeeds."""

    primary_llm = AsyncMock()
    primary_llm.chat_completion.return_value = "primary summary"

    local_llm = AsyncMock()

    engine = LocalFirstContextEngine(llm=primary_llm, local_llm=local_llm)

    entry1 = MagicMock(spec=MemoryEntry)
    entry1.content = "content1"

    entry2 = MagicMock(spec=MemoryEntry)
    entry2.content = "content2"

    items = [entry1, entry2]
    # Limit is 1, so compaction will trigger
    compacted = await engine.compact(items, {"limit": 1})

    assert primary_llm.chat_completion.called
    assert not local_llm.chat_completion.called

    assert len(compacted) == 1
    assert compacted[0].content == "primary summary"
