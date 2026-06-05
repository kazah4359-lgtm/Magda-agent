import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.context.engine import ContextEngine
from magda_agent.context.plugin import ContextPlugin
from magda_agent.context.default_plugin import DefaultContextPlugin
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState

class MockPlugin(ContextPlugin):
    def __init__(self):
        self.bootstrap_called = False
        self.ingest_called = False
        self.assemble_called = False
        self.compact_called = False

    async def bootstrap(self, config):
        self.bootstrap_called = True

    async def ingest(self, content, metadata):
        self.ingest_called = True
        return f"ingested_{content}"

    async def assemble(self, context_items, metadata):
        self.assemble_called = True
        return "assembled_context"

    async def compact(self, context_items, metadata):
        self.compact_called = True
        return context_items[:-1]

@pytest.mark.asyncio
async def test_context_engine_lifecycle():
    mock_plugin = MockPlugin()
    engine = ContextEngine(plugins=[mock_plugin])

    # Test Bootstrap
    await engine.bootstrap_all({"key": "value"})
    assert mock_plugin.bootstrap_called

    # Test Ingest
    ingested = await engine.ingest("raw", {"user_id": 1})
    assert mock_plugin.ingest_called
    assert ingested == "ingested_raw"

    # Test Assemble
    assembled = await engine.assemble([], {"user_id": 1})
    assert mock_plugin.assemble_called
    assert assembled == "assembled_context"

    # Test Compact
    items = [1, 2, 3]
    compacted = await engine.compact(items, {"limit": 2})
    assert mock_plugin.compact_called
    assert len(compacted) == 2

@pytest.mark.asyncio
async def test_default_plugin_assemble():
    plugin = DefaultContextPlugin()
    entry = MagicMock(spec=MemoryEntry)
    entry.content = "hello"

    assembled = await plugin.assemble([entry], {})
    assert assembled == "- hello"

@pytest.mark.asyncio
async def test_default_plugin_compact_no_llm():
    plugin = DefaultContextPlugin(llm=None)
    entry1 = MagicMock(spec=MemoryEntry)
    entry2 = MagicMock(spec=MemoryEntry)

    # Limit is 1, we have 2 items
    items = [entry1, entry2]
    compacted = await plugin.compact(items, {"limit": 1})

    # Should drop the first item
    assert len(compacted) == 1
    assert compacted[0] == entry2

@pytest.mark.asyncio
async def test_default_plugin_compact_with_llm():
    llm = AsyncMock()
    llm.chat_completion.return_value = "summary"
    plugin = DefaultContextPlugin(llm=llm)

    entry1 = MagicMock(spec=MemoryEntry)
    entry1.content = "content1"
    entry1.importance = 0.5
    entry2 = MagicMock(spec=MemoryEntry)
    entry2.content = "content2"
    entry2.importance = 0.5

    items = [entry1, entry2]
    # Limit 1, items 2 -> compact
    compacted = await plugin.compact(items, {"limit": 1})

    assert len(compacted) == 1
    assert compacted[0].content == "summary"
    assert llm.chat_completion.called
