import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.context_engine import ContextEngine, ContextPlugin
from magda_agent.memory.default_context_plugin import DefaultContextPlugin
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

    def before_retrieval(self, query: str, user_id: int) -> str:
        return query

    def after_retrieval(self, context: list, query: str, user_id: int) -> list:
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        pass

@pytest.mark.asyncio
async def test_context_engine_lifecycle():
    mock_plugin = MockPlugin()
    engine = ContextEngine(plugins=[mock_plugin])

    # Test Bootstrap
    config = {"key": "value"}
    # Capture the config passed to the plugin's bootstrap method
    bootstrap_config = None

    async def capture_bootstrap(cfg):
        nonlocal bootstrap_config
        bootstrap_config = cfg
        mock_plugin.bootstrap_called = True

    mock_plugin.bootstrap = capture_bootstrap

    await engine.bootstrap_all(config)
    assert mock_plugin.bootstrap_called
    # Config should remain unchanged at caller level, but inner config had hook_registry
    assert "key" in config
    assert "hook_registry" not in config
    assert bootstrap_config is not None
    assert "hook_registry" in bootstrap_config
    assert bootstrap_config["hook_registry"] == engine.hook_registry

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

def test_context_engine_retrieve_context_hooks() -> None:
    """Test before_retrieval and after_retrieval hooks explicitly."""
    plugin = MockPlugin()
    # Ensure it modifies
    plugin.before_retrieval = MagicMock(return_value="modified_query")
    plugin.after_retrieval = MagicMock(return_value=["modified_context"])

    engine = ContextEngine(plugins=[plugin])

    base_retrieval = MagicMock(return_value=["base_context"])

    result = engine.retrieve_context("query", 1, base_retrieval)

    plugin.before_retrieval.assert_called_once_with("query", 1)
    base_retrieval.assert_called_once_with("modified_query", 1)
    plugin.after_retrieval.assert_called_once_with(["base_context"], "modified_query", 1)

    assert result == ["modified_context"]

def test_context_engine_hook_registry_integration() -> None:
    """Test that HookRegistry is initialized and plugins can register hooks via it."""
    engine = ContextEngine()

    assert hasattr(engine, "hook_registry")

    mock_hook = MagicMock(return_value="hooked")
    engine.hook_registry.register_hook("custom_hook", mock_hook)

    result = engine.hook_registry.trigger_hook("custom_hook", "arg1")
    assert result == "hooked"
    mock_hook.assert_called_once_with("arg1")

@pytest.mark.asyncio
async def test_context_engine_builtin_compaction() -> None:
    """Tests that the ContextEngine built-in fallback compression correctly summarizes context items using the LLM when no plugins act on it."""
    llm = AsyncMock()
    llm.chat_completion.return_value = "engine summary"

    # Engine with no plugins but with an LLM
    engine = ContextEngine(llm=llm)

    entry1 = MagicMock(spec=MemoryEntry)
    entry1.content = "content1"
    entry1.importance = 0.5
    entry2 = MagicMock(spec=MemoryEntry)
    entry2.content = "content2"
    entry2.importance = 0.5

    items = [entry1, entry2]
    # Limit 1, items 2 -> compact via engine fallback
    compacted = await engine.compact(items, {"limit": 1})

    assert len(compacted) == 1
    assert compacted[0].content == "engine summary"
    assert llm.chat_completion.called

@pytest.mark.asyncio
async def test_context_engine_builtin_compaction_no_llm() -> None:
    """Tests that the ContextEngine built-in fallback compression gracefully drops the oldest item when no LLM is provided."""
    # Engine with no plugins and no LLM
    engine = ContextEngine()

    entry1 = MagicMock(spec=MemoryEntry)
    entry2 = MagicMock(spec=MemoryEntry)

    items = [entry1, entry2]
    # Limit 1, items 2 -> compact by dropping oldest item
    compacted = await engine.compact(items, {"limit": 1})

    assert len(compacted) == 1
    assert compacted[0] == entry2

@pytest.mark.asyncio
async def test_context_engine_advanced_compaction_tags() -> None:
    """Tests that the ContextEngine fallback compression merges tags from multiple entries."""
    llm = AsyncMock()
    llm.chat_completion.return_value = "advanced summary"
    engine = ContextEngine(llm=llm)

    entry1 = MagicMock(spec=MemoryEntry)
    entry1.content = "content1"
    entry1.importance = 0.5
    entry1.tags = ["tag1", "tag2"]

    entry2 = MagicMock(spec=MemoryEntry)
    entry2.content = "content2"
    entry2.importance = 0.5
    entry2.tags = ["tag2", "tag3"]

    items = [entry1, entry2]
    compacted = await engine.compact(items, {"limit": 1})

    assert len(compacted) == 1
    assert compacted[0].content == "advanced summary"
    assert set(compacted[0].tags) == {"tag1", "tag2", "tag3"}

    # Also verify prompt contains new instructions
    call_args = llm.chat_completion.call_args[0][0]
    user_prompt = call_args[1]["content"]
    assert "maintaining key facts and semantic links" in user_prompt

def test_context_engine_write_context_hooks() -> None:
    """Test before_write and after_write hooks explicitly."""
    plugin = MockPlugin()
    plugin.before_write = MagicMock(return_value="modified_context")
    plugin.after_write = MagicMock()
    plugin.on_context_update = MagicMock()

    engine = ContextEngine(plugins=[plugin])

    engine.write_context("base_context", 1)

    plugin.before_write.assert_called_once_with("base_context", 1)
    plugin.on_context_update.assert_called_once_with("modified_context", 1)
    plugin.after_write.assert_called_once_with("modified_context", 1)
