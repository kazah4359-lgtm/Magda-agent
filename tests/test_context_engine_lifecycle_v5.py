import pytest
import pytest_asyncio
from typing import Dict, Any, List
from magda_agent.memory.context_engine_lifecycle_v5 import ContextPluginLifecycleV5, ContextEngineLifecycleV5

class MockLifecyclePlugin(ContextPluginLifecycleV5):
    """A mock plugin to test lifecycle hooks."""
    def __init__(self):
        self.loaded = False
        self.unloaded = False
        self.config = {}

    async def on_load(self, config: Dict[str, Any]) -> None:
        self.loaded = True
        self.config = config

    async def on_unload(self) -> None:
        self.unloaded = True

    def before_write(self, context: Any, user_id: int) -> Any:
        return context

@pytest_asyncio.fixture
async def context_engine():
    """Fixture to provide a ContextEngineLifecycleV5 instance."""
    engine = ContextEngineLifecycleV5()
    yield engine

@pytest.mark.asyncio
async def test_load_plugin(context_engine):
    """Test that a plugin is correctly loaded and its on_load hook is called."""
    plugin = MockLifecyclePlugin()
    config = {"key": "value"}

    assert not plugin.loaded
    await context_engine.load_plugin(plugin, config)

    assert plugin.loaded
    assert plugin.config == config
    assert plugin in context_engine._plugins

@pytest.mark.asyncio
async def test_unload_plugin(context_engine):
    """Test that a plugin is correctly unloaded and its on_unload hook is called."""
    plugin = MockLifecyclePlugin()

    await context_engine.load_plugin(plugin)
    assert plugin in context_engine._plugins
    assert not plugin.unloaded

    await context_engine.unload_plugin(plugin)

    assert plugin.unloaded
    assert plugin not in context_engine._plugins

@pytest.mark.asyncio
async def test_unload_unregistered_plugin(context_engine, caplog):
    """Test that unloading an unregistered plugin logs a warning and doesn't crash."""
    plugin = MockLifecyclePlugin()

    await context_engine.unload_plugin(plugin)
    assert "is not registered" in caplog.text

@pytest.mark.asyncio
async def test_hook_deregistration_on_unload(context_engine):
    """Test that plugin hooks are removed from the registry when unloaded."""
    plugin = MockLifecyclePlugin()
    await context_engine.load_plugin(plugin)

    # Verify the hook is registered
    assert len(context_engine.hook_registry._hooks.get('before_write', [])) > 0

    await context_engine.unload_plugin(plugin)

    # Verify the hook is deregistered
    assert len(context_engine.hook_registry._hooks.get('before_write', [])) == 0
