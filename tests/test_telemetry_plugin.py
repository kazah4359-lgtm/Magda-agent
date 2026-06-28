import pytest
from unittest.mock import MagicMock
from collections import deque
from magda_agent.architecture.telemetry_plugin import TelemetryPlugin
from magda_agent.memory.context_engine import ContextEngine

@pytest.fixture
def telemetry_plugin() -> TelemetryPlugin:
    """Fixture providing a TelemetryPlugin instance."""
    return TelemetryPlugin()

@pytest.fixture
def context_engine(telemetry_plugin: TelemetryPlugin) -> ContextEngine:
    """Fixture providing a ContextEngine instance with a TelemetryPlugin registered."""
    return ContextEngine(plugins=[telemetry_plugin])

def test_telemetry_plugin_initialization(telemetry_plugin: TelemetryPlugin) -> None:
    """Test that TelemetryPlugin initializes correctly with deques."""
    assert telemetry_plugin.metrics["before_retrieval_calls"] == 0
    assert telemetry_plugin.metrics["after_retrieval_calls"] == 0
    assert telemetry_plugin.metrics["total_retrieved_items"] == 0
    assert isinstance(telemetry_plugin.metrics["queries"], deque)
    assert telemetry_plugin.metrics["queries"].maxlen == 1000
    assert isinstance(telemetry_plugin.metrics["retrieval_times"], deque)
    assert telemetry_plugin.metrics["bootstrap_count"] == 0
    assert telemetry_plugin.metrics["ingest_count"] == 0
    assert telemetry_plugin._start_time == 0.0

def test_telemetry_plugin_hooks_direct(telemetry_plugin: TelemetryPlugin) -> None:
    """Test TelemetryPlugin hooks directly."""
    query = "test query"
    user_id = 1

    # Trigger before_retrieval
    ret_query = telemetry_plugin.before_retrieval(query, user_id)
    assert ret_query == query
    assert telemetry_plugin.metrics["before_retrieval_calls"] == 1
    assert query in telemetry_plugin.metrics["queries"]
    assert telemetry_plugin._start_time > 0.0

    # Trigger after_retrieval
    context = ["item1", "item2"]
    ret_context = telemetry_plugin.after_retrieval(context, query, user_id)
    assert ret_context == context
    assert telemetry_plugin.metrics["after_retrieval_calls"] == 1
    assert telemetry_plugin.metrics["total_retrieved_items"] == 2
    assert len(telemetry_plugin.metrics["retrieval_times"]) == 1
    assert telemetry_plugin.metrics["retrieval_times"][0] >= 0.0

def test_telemetry_plugin_integration_with_context_engine(
    context_engine: ContextEngine, telemetry_plugin: TelemetryPlugin
) -> None:
    """Test TelemetryPlugin integration via ContextEngine's hook registry."""
    query = "integrated query"
    user_id = 2
    mock_retrieval_func = MagicMock(return_value=["res1", "res2", "res3"])

    # Retrieve context, which should trigger hooks
    result = context_engine.retrieve_context(query, user_id, mock_retrieval_func)

    # Assert return values
    assert result == ["res1", "res2", "res3"]
    mock_retrieval_func.assert_called_once_with(query, user_id)

    # Assert metrics
    assert telemetry_plugin.metrics["before_retrieval_calls"] == 1
    assert query in telemetry_plugin.metrics["queries"]
    assert telemetry_plugin.metrics["after_retrieval_calls"] == 1
    assert telemetry_plugin.metrics["total_retrieved_items"] == 3
    assert len(telemetry_plugin.metrics["retrieval_times"]) == 1

@pytest.mark.asyncio
async def test_telemetry_plugin_other_hooks(telemetry_plugin: TelemetryPlugin) -> None:
    """Test other protocol methods and legacy metrics behave correctly."""
    await telemetry_plugin.bootstrap({})
    assert telemetry_plugin.metrics["bootstrap_count"] == 1

    assert await telemetry_plugin.ingest("content", {}) == "content"
    assert telemetry_plugin.metrics["ingest_count"] == 1

    assert await telemetry_plugin.assemble(["1", "2"], {}) == "1\n2"
    assert telemetry_plugin.metrics["assemble_count"] == 1

    assert await telemetry_plugin.compact(["1"], {}) == ["1"]
    assert telemetry_plugin.metrics["compact_count"] == 1

    telemetry_plugin.on_context_update("context", 1)
    assert telemetry_plugin.metrics["on_context_update_count"] == 1

    assert telemetry_plugin.before_write("context", 1) == "context"
