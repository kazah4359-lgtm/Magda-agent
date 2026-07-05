import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.context_engine_v1 import ContextEngineV1, ContextPluginV1

class DummyPlugin(ContextPluginV1):
    def __init__(self):
        self.bootstrap_called = False
        self.ingest_called = False
        self.assemble_called = False
        self.compact_called = False
        self.before_retrieval_called = False
        self.after_retrieval_called = False
        self.before_write_called = False
        self.after_write_called = False
        self.on_context_update_called = False

    async def bootstrap(self, config):
        self.bootstrap_called = True

    async def ingest(self, content, metadata):
        self.ingest_called = True
        return content + " ingested"

    async def assemble(self, context_items, metadata):
        self.assemble_called = True
        return " assembled"

    async def compact(self, context_items, metadata):
        self.compact_called = True
        return context_items

    def before_retrieval(self, query, user_id):
        self.before_retrieval_called = True
        return query + " modified"

    def after_retrieval(self, context, query, user_id):
        self.after_retrieval_called = True
        return context + ["added"]

    def before_write(self, context, user_id):
        self.before_write_called = True
        return context

    def after_write(self, context, user_id):
        self.after_write_called = True

    def on_context_update(self, new_context, user_id):
        self.on_context_update_called = True

@pytest.mark.asyncio
async def test_context_engine_v1_hooks():
    plugin = DummyPlugin()
    engine = ContextEngineV1(plugins=[plugin])

    await engine.bootstrap_all({"test": "config"})
    assert plugin.bootstrap_called

    res_ingest = await engine.ingest("test", {})
    assert plugin.ingest_called
    assert res_ingest == "test ingested"

    res_assemble = await engine.assemble(["item1"], {})
    assert plugin.assemble_called
    assert res_assemble == " assembled"

    res_compact = await engine.compact(["item1"], {})
    assert plugin.compact_called
    assert res_compact == ["item1"]

    base_retrieval = MagicMock(return_value=["item1"])
    res_retrieval = engine.retrieve_context("query", 1, base_retrieval)
    assert plugin.before_retrieval_called
    assert plugin.after_retrieval_called
    assert res_retrieval == ["item1", "added"]
    base_retrieval.assert_called_with("query modified", 1)

    engine.write_context("context_data", 1)
    assert plugin.before_write_called
    assert plugin.on_context_update_called
    assert plugin.after_write_called

@pytest.mark.asyncio
async def test_context_engine_v1_fallback_compaction():
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = "compressed_test"

    engine = ContextEngineV1(llm=mock_llm)
    items = ["item1", "item2", "item3"]
    res = await engine.compact(items, {"limit": 2})

    assert len(res) == 2
    assert res[0].content == "compressed_test"
    assert res[1] == "item3"
    mock_llm.chat_completion.assert_called_once()
