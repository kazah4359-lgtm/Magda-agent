import pytest
from unittest.mock import MagicMock
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.context_engine_hooks_v5 import ContextPluginV5

def test_context_plugin_v5_hooks_execution() -> None:
    """
    Test that ContextEngine correctly invokes ContextPluginV5 hooks
    during context retrieval.
    """
    plugin = ContextPluginV5()
    engine = ContextEngine(plugins=[plugin])

    # Mock base retrieval function
    mock_base_retrieval = MagicMock(return_value=["item1", "item2"])

    # Execute retrieval
    user_id = 123
    query = "test query"
    result = engine.retrieve_context(query, user_id, mock_base_retrieval)

    # Verify before_retrieval hook modified the query
    expected_modified_query = f"{query} [v5_refined]"
    mock_base_retrieval.assert_called_once_with(expected_modified_query, user_id)

    # Verify after_retrieval hook modified the context
    assert len(result) == 3
    assert result[0] == "item1"
    assert result[1] == "item2"
    assert result[2] == f"v5_augmented_for_{user_id}"

    # Verify execution log
    assert "before_retrieval" in plugin.hooks_log
    assert "after_retrieval" in plugin.hooks_log
    assert plugin.hooks_log.index("before_retrieval") < plugin.hooks_log.index("after_retrieval")

def test_context_plugin_v5_update_hook() -> None:
    """Test that ContextEngine correctly invokes on_context_update hook."""
    plugin = ContextPluginV5()
    engine = ContextEngine(plugins=[plugin])

    user_id = 456
    new_data = "new context data"
    engine.update_context(new_data, user_id)

    assert "on_context_update" in plugin.hooks_log

@pytest.mark.asyncio
async def test_context_plugin_v5_async_lifecycle() -> None:
    """Test async lifecycle methods of ContextPluginV5."""
    plugin = ContextPluginV5()

    await plugin.bootstrap({"config": "test"})
    assert "bootstrap" in plugin.hooks_log

    await plugin.ingest("raw content", {})
    assert "ingest" in plugin.hooks_log

    assembled = await plugin.assemble(["item"], {})
    assert "assemble" in plugin.hooks_log
    assert assembled == "item"

    compacted = await plugin.compact(["item1", "item2"], {"limit": 1})
    assert "compact" in plugin.hooks_log
    assert len(compacted) == 2
