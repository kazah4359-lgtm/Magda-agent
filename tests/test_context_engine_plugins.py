from typing import Any
import pytest
from unittest.mock import MagicMock
from magda_agent.memory.context_engine import ContextEngine, ContextPlugin

class MockPlugin:
    def before_retrieval(self, query: str, user_id: int) -> str:
        return query + " [plugin_modified]"

    def after_retrieval(self, context: list, query: str, user_id: int) -> list:
        return context + ["plugin_appended"]

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        pass


def test_register_plugin() -> None:
    """Test registering a plugin."""
    engine = ContextEngine()
    plugin = MockPlugin()
    engine.register_plugin(plugin)
    assert len(engine._plugins) == 1
    assert engine._plugins[0] == plugin


def test_retrieve_context() -> None:
    """Test context retrieval hooks."""
    engine = ContextEngine()
    plugin = MockPlugin()
    engine.register_plugin(plugin)

    mock_base_retrieval = MagicMock(return_value=["base_context"])

    result = engine.retrieve_context("initial query", 1, mock_base_retrieval)

    mock_base_retrieval.assert_called_once_with("initial query [plugin_modified]", 1)
    assert result == ["base_context", "plugin_appended"]


def test_update_context() -> None:
    """Test context update hooks."""
    engine = ContextEngine()
    plugin = MagicMock(spec=ContextPlugin)
    engine.register_plugin(plugin)

    engine.update_context("new_data", 1)
    plugin.on_context_update.assert_called_once_with("new_data", 1)


@pytest.mark.asyncio
async def test_memory_system_context_integration():
    """Test MemorySystem's interaction with ContextEngine hooks."""
    from magda_agent.memory.storage import MemorySystem
    from magda_agent.emotions.engine import PADState

    engine = ContextEngine()
    plugin = MagicMock(spec=ContextPlugin)
    engine.register_plugin(plugin)

    # Set up return values for hooks that expect a return
    plugin.before_retrieval.side_effect = lambda q, uid: q + "_modified"
    plugin.after_retrieval.side_effect = lambda ctx, q, uid: ctx + ["appended"]

    mem_system = MemorySystem(context_engine=engine)

    # 1. Test on_context_update hook via add_memory
    pad = PADState(0, 0, 0)
    await mem_system.add_memory("test content", 0.5, pad, user_id=1)

    assert plugin.on_context_update.called
    # The first argument should be a MemoryEntry
    args, kwargs = plugin.on_context_update.call_args
    assert args[0].content == "test content"
    assert args[1] == 1

    # 2. Test before_retrieval and after_retrieval hooks via retrieve_relevant
    # We need some entries in working memory for base_retrieval to return something
    # Actually, base_retrieval returns empty if no entries match, but we can still check the hook calls
    result = mem_system.retrieve_relevant("query", user_id=1)

    assert plugin.before_retrieval.called
    plugin.before_retrieval.assert_called_with("query", 1)

    assert plugin.after_retrieval.called
    # result should contain "appended" from the hook
    assert "appended" in result
