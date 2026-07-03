import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.memory.context_plugin_v4 import ContextPluginV4

class DummyEntry:
    def __init__(self, content):
        self.content = content

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Summarized v4 content")
    return llm

@pytest.fixture
def plugin(mock_llm):
    return ContextPluginV4(llm=mock_llm)

@pytest.mark.asyncio
async def test_plugin_bootstrap(plugin):
    await plugin.bootstrap({"test_key": "test_val"})
    assert plugin.config == {"test_key": "test_val"}

@pytest.mark.asyncio
async def test_plugin_ingest(plugin):
    content = "hello world"
    res = await plugin.ingest(content, {"user_id": "123"})
    assert res == "[V4:123] hello world"

@pytest.mark.asyncio
async def test_plugin_assemble(plugin):
    items = [DummyEntry("item1"), DummyEntry("item2")]
    res = await plugin.assemble(items, {})
    assert "--- OpenClaw V4 Context Engine ---" in res
    assert "- item1" in res
    assert "- item2" in res
    assert "----------------------------------" in res

@pytest.mark.asyncio
async def test_plugin_assemble_empty(plugin):
    res = await plugin.assemble([], {})
    assert res == ""

@pytest.mark.asyncio
async def test_plugin_compact_under_limit(plugin):
    items = [DummyEntry("item1"), DummyEntry("item2")]
    res = await plugin.compact(items, {"limit": 5})
    assert len(res) == 2

@pytest.mark.asyncio
async def test_plugin_compact_over_limit_with_llm(plugin):
    items = [DummyEntry(f"item{i}") for i in range(5)]
    res = await plugin.compact(items, {"limit": 3})
    assert len(res) == 4  # 2 compressed into 1, plus 3 remaining
    assert res[0].content == "Summarized v4 content"
    assert res[0].tags == ["v4_compacted"]

@pytest.mark.asyncio
async def test_plugin_compact_over_limit_no_llm():
    plugin_no_llm = ContextPluginV4(llm=None)
    items = [DummyEntry(f"item{i}") for i in range(5)]
    res = await plugin_no_llm.compact(items, {"limit": 3})
    assert len(res) == 3
    assert res[0].content == "item2"

def test_plugin_before_retrieval(plugin):
    res = plugin.before_retrieval("search query", 1)
    assert res == "search query (v4 enhanced for user 1)"

def test_plugin_after_retrieval(plugin):
    res = plugin.after_retrieval(["existing"], "query", 1)
    assert len(res) == 2
    assert res[1] == "V4 System Note: Retrieved for query 'query'"

def test_plugin_on_context_update(plugin):
    # Should not raise exception
    plugin.on_context_update("new context", 1)

def test_plugin_before_after_write(plugin):
    res = plugin.before_write("context_item", 1)
    assert res == "context_item"
    plugin.after_write("context_item", 1) # Should not raise
