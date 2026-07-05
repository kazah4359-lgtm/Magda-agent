import pytest
from typing import Dict, Any
from magda_agent.memory.context_engine_lifecycle_v2 import ContextEngineLifecycleV2, ContextPluginLifecycleV2

class DummyLifecyclePlugin(ContextPluginLifecycleV2):
    def __init__(self):
        self.on_load_called = False
        self.on_unload_called = False
        self.config_received = None
        self.before_retrieval_called = False

    async def on_load(self, config: Dict[str, Any]) -> None:
        self.on_load_called = True
        self.config_received = config

    async def on_unload(self) -> None:
        self.on_unload_called = True

    def before_retrieval(self, query: str, user_id: int) -> str:
        """Modify query before retrieval."""
        self.before_retrieval_called = True
        return query + " modified"

@pytest.mark.asyncio
async def test_lifecycle_hooks() -> None:
    """Test load and unload hooks correctly manipulate the plugin list and registry."""
    plugin = DummyLifecyclePlugin()
    engine = ContextEngineLifecycleV2()

    # Load plugin
    await engine.load_plugin(plugin, {"test_key": "test_value"})
    assert plugin.on_load_called is True
    assert plugin.config_received == {"test_key": "test_value"}
    assert plugin in engine._plugins

    # Test hook registry was updated
    res = engine.retrieve_context("query", 1, lambda q, u: [q])
    assert plugin.before_retrieval_called is True
    assert res == ["query modified"]

    # Unload plugin
    await engine.unload_plugin(plugin)
    assert plugin.on_unload_called is True
    assert plugin not in engine._plugins

    # Verify hooks are removed
    plugin.before_retrieval_called = False # reset
    res = engine.retrieve_context("query", 1, lambda q, u: [q])
    assert plugin.before_retrieval_called is False
    assert res == ["query"]
