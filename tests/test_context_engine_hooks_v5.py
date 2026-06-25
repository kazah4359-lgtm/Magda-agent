import pytest
from typing import List, Any
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.context_engine_hooks_v5 import OrderedContextPluginV5

def dummy_retrieval_func(query: str, user_id: int) -> List[Any]:
    """A dummy base retrieval function for testing context retrieval."""
    return [f"retrieved for: {query}"]

@pytest.mark.asyncio
async def test_ordered_context_plugin_v5_async_methods() -> None:
    """Test the asynchronous lifecycle methods of OrderedContextPluginV5."""
    plugin = OrderedContextPluginV5()

    await plugin.bootstrap({"key": "value"})
    assert plugin.execution_order == ["bootstrap"]

    await plugin.ingest("content", {})
    assert plugin.execution_order == ["bootstrap", "ingest"]

    await plugin.assemble([], {})
    assert plugin.execution_order == ["bootstrap", "ingest", "assemble"]

    await plugin.compact([], {})
    assert plugin.execution_order == ["bootstrap", "ingest", "assemble", "compact"]

def test_context_engine_hooks_v5_execution_order() -> None:
    """Test the execution order of retrieval hooks in ContextEngine."""
    plugin = OrderedContextPluginV5()
    engine = ContextEngine(plugins=[plugin])

    result = engine.retrieve_context("test query", 42, dummy_retrieval_func)

    # Check modifications
    assert len(result) == 2
    assert "test query [v5_pre_retrieval_modified]" in result[0]
    assert "metadata: v5_post_retrieval executed for user 42" in result[1]

    # Check execution order
    assert plugin.execution_order == ["before_retrieval", "after_retrieval"]

def test_context_engine_hooks_v5_update_context() -> None:
    """Test the update_context hook triggers correctly."""
    plugin = OrderedContextPluginV5()
    engine = ContextEngine(plugins=[plugin])

    engine.update_context("new_context_data", 42)
    assert plugin.execution_order == ["on_context_update"]
