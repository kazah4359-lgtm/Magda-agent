import pytest
from typing import List, Any
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.context_engine_hooks_v2 import OrderedContextPluginV2

def dummy_retrieval_func(query: str, user_id: int) -> List[Any]:
    return [f"retrieved for: {query}"]

@pytest.mark.asyncio
async def test_ordered_context_plugin_v2_async_methods() -> None:
    plugin = OrderedContextPluginV2()

    await plugin.bootstrap({"key": "value"})
    assert plugin.execution_order == ["bootstrap"]

    await plugin.ingest("content", {})
    assert plugin.execution_order == ["bootstrap", "ingest"]

    await plugin.assemble([], {})
    assert plugin.execution_order == ["bootstrap", "ingest", "assemble"]

    await plugin.compact([], {})
    assert plugin.execution_order == ["bootstrap", "ingest", "assemble", "compact"]

def test_context_engine_hooks_v2_execution_order() -> None:
    plugin = OrderedContextPluginV2()
    engine = ContextEngine(plugins=[plugin])

    result = engine.retrieve_context("test query", 42, dummy_retrieval_func)

    # Check modifications
    assert len(result) == 2
    assert "test query [v2_pre_retrieval_modified]" in result[0]
    assert "metadata: v2_post_retrieval executed for user 42" in result[1]

    # Check execution order
    assert plugin.execution_order == ["before_retrieval", "after_retrieval"]

def test_context_engine_hooks_v2_update_context() -> None:
    plugin = OrderedContextPluginV2()
    engine = ContextEngine(plugins=[plugin])

    engine.update_context("new_context_data", 42)
    assert plugin.execution_order == ["on_context_update"]
