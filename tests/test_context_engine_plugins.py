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
