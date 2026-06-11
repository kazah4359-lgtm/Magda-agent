import pytest
from typing import List, Any
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.memory.context_engine_lifecycle import LifecyclePlugin

def dummy_retrieval_func(query: str, user_id: int) -> List[Any]:
    return [f"retrieved for: {query}"]

def test_context_engine_lifecycle_hooks():
    plugin = LifecyclePlugin()
    engine = ContextEngine(plugins=[plugin])

    result = engine.retrieve_context("test query", 1, dummy_retrieval_func)

    assert len(result) == 2
    assert "test query (modified by pre_retrieval hook)" in result[0]
    assert "metadata: post_retrieval hook executed for 1" in result[1]
