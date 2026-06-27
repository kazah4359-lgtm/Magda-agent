import pytest
from magda_agent.architecture.telemetry_plugin import TelemetryPlugin


@pytest.mark.asyncio
async def test_telemetry_plugin_hooks():
    plugin = TelemetryPlugin()

    # Check initial state
    assert plugin.metrics["before_retrieval_count"] == 0
    assert plugin.metrics["after_retrieval_count"] == 0
    assert plugin.metrics["total_context_items_retrieved"] == 0

    # Test before_retrieval
    query = "test query"
    user_id = 1
    result_query = plugin.before_retrieval(query, user_id)

    assert result_query == query
    assert plugin.metrics["before_retrieval_count"] == 1

    # Test after_retrieval
    context = ["item1", "item2"]
    result_context = plugin.after_retrieval(context, query, user_id)

    assert result_context == context
    assert plugin.metrics["after_retrieval_count"] == 1
    assert plugin.metrics["total_context_items_retrieved"] == 2

    # Call after_retrieval again
    context2 = ["item3"]
    plugin.after_retrieval(context2, query, user_id)

    assert plugin.metrics["after_retrieval_count"] == 2
    assert plugin.metrics["total_context_items_retrieved"] == 3


@pytest.mark.asyncio
async def test_telemetry_plugin_other_hooks():
    plugin = TelemetryPlugin()

    # Test bootstrap
    await plugin.bootstrap({"config_key": "value"})
    assert plugin.metrics["bootstrap_count"] == 1

    # Test ingest
    content = "test content"
    result_content = await plugin.ingest(content, {})
    assert result_content == content
    assert plugin.metrics["ingest_count"] == 1

    # Test assemble
    context = ["item1", "item2"]
    result_assemble = await plugin.assemble(context, {})
    assert result_assemble == "item1\nitem2"
    assert plugin.metrics["assemble_count"] == 1

    # Test compact
    result_compact = await plugin.compact(context, {})
    assert result_compact == context
    assert plugin.metrics["compact_count"] == 1

    # Test on_context_update
    plugin.on_context_update("new_context", 1)
    assert plugin.metrics["on_context_update_count"] == 1
