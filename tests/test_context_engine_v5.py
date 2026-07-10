import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Any
from magda_agent.memory.context_engine_v5 import ContextEngineV5, DynamicHookRegistry
from magda_agent.memory.context_engine import ContextPlugin

class MockDynamicPlugin(ContextPlugin):
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.bootstrap_called = False
        self.ingest_called = False
        self.assemble_called = False
        self.compact_called = False

    async def bootstrap(self, config):
        self.bootstrap_called = True

    async def ingest(self, content, metadata):
        self.ingest_called = True
        return f"{self.prefix}ingested_{content}"

    async def assemble(self, context_items, metadata):
        self.assemble_called = True
        return f"{self.prefix}assembled_context"

    async def compact(self, context_items, metadata):
        self.compact_called = True
        return context_items[:-1] if context_items else []

    def before_retrieval(self, query: str, user_id: int) -> str:
        return f"{self.prefix}{query}"

    def after_retrieval(self, context: list, query: str, user_id: int) -> list:
        return context

    def before_write(self, context: Any, user_id: int) -> Any:
        return context

    def after_write(self, context: Any, user_id: int) -> None:
        pass

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        pass


def test_dynamic_hook_registry_unregister() -> None:
    """Test that dynamic hook registry can unregister hooks."""
    registry = DynamicHookRegistry()

    mock_func = MagicMock(return_value="hooked")

    # Test registration
    registry.register_hook("custom_hook", mock_func)
    assert len(registry._hooks["custom_hook"]) == 1

    # Test execution
    result = registry.trigger_hook("custom_hook", "arg")
    assert result == "hooked"
    mock_func.assert_called_once_with("arg")

    # Test unregistration
    registry.unregister_hook("custom_hook", mock_func)
    assert len(registry._hooks["custom_hook"]) == 0

    # Test execution after unregistration
    result_after = registry.trigger_hook("custom_hook", "arg2")
    assert result_after == "arg2" # Fallback behavior returns first arg
    assert mock_func.call_count == 1 # Still 1, wasn't called again


@pytest.mark.asyncio
async def test_context_engine_v5_dynamic_plugin_registration() -> None:
    """Test that ContextEngineV5 can dynamically register and unregister plugins."""
    engine = ContextEngineV5()

    plugin1 = MockDynamicPlugin(prefix="p1_")
    plugin2 = MockDynamicPlugin(prefix="p2_")

    # Register plugin1
    engine.register_plugin(plugin1)

    # Verify plugin1 hooks are active
    assert engine.hook_registry.trigger_hook('before_retrieval', "query", 1) == "p1_query"
    assert len(engine._plugins) == 1

    # Register plugin2
    engine.register_plugin(plugin2)
    assert len(engine._plugins) == 2

    # Verify chain of hooks.
    # 'query' -> p1_before_retrieval -> 'p1_query' -> p2_before_retrieval -> 'p2_p1_query'
    assert engine.hook_registry.trigger_hook('before_retrieval', "query", 1) == "p2_p1_query"

    # Unregister plugin1
    engine.unregister_plugin(plugin1)
    assert len(engine._plugins) == 1
    assert plugin1 not in engine._plugins

    # Verify plugin1 hooks are gone, plugin2 hooks are active
    # 'query' -> p2_before_retrieval -> 'p2_query'
    assert engine.hook_registry.trigger_hook('before_retrieval', "query", 1) == "p2_query"

    # Unregister plugin2
    engine.unregister_plugin(plugin2)
    assert len(engine._plugins) == 0

    # Verify all hooks are gone
    assert engine.hook_registry.trigger_hook('before_retrieval', "query", 1) == "query"

def test_unregister_unregistered_plugin() -> None:
    """Test unregistering a plugin that is not registered."""
    engine = ContextEngineV5()
    plugin1 = MockDynamicPlugin()

    # Should not raise any errors, just log a warning
    engine.unregister_plugin(plugin1)
    assert len(engine._plugins) == 0

def test_unregister_unregistered_hook() -> None:
    """Test unregistering a hook that is not registered."""
    registry = DynamicHookRegistry()
    mock_func = MagicMock()

    # Should not raise any errors, just log a warning
    registry.unregister_hook("nonexistent_hook", mock_func)

    registry.register_hook("existing_hook", mock_func)
    mock_func2 = MagicMock()
    # Unregistering a function that wasn't registered for this hook
    registry.unregister_hook("existing_hook", mock_func2)
    assert len(registry._hooks["existing_hook"]) == 1
