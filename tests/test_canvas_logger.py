import pytest
import asyncio
from typing import Dict, Any, List
from magda_agent.visualization.canvas_logger import CanvasLoggerPlugin

def test_canvas_logger_bootstrap():
    plugin = CanvasLoggerPlugin()
    config = {"setting": "value"}

    # Run async function using asyncio
    asyncio.run(plugin.bootstrap(config))

    logs = plugin.get_logs()
    assert len(logs) == 1
    assert logs[0]["event"] == "bootstrap"
    assert logs[0]["config"] == config

def test_canvas_logger_before_retrieval():
    plugin = CanvasLoggerPlugin()
    query = "test query"
    user_id = 42

    result = plugin.before_retrieval(query, user_id)

    assert result == query
    logs = plugin.get_logs()
    assert len(logs) == 1
    assert logs[0]["event"] == "before_retrieval"
    assert logs[0]["query"] == query
    assert logs[0]["user_id"] == user_id

def test_canvas_logger_after_retrieval():
    plugin = CanvasLoggerPlugin()
    context = ["item1", "item2", "item3"]
    query = "test query"
    user_id = 42

    result = plugin.after_retrieval(context, query, user_id)

    assert result == context
    logs = plugin.get_logs()
    assert len(logs) == 1
    assert logs[0]["event"] == "after_retrieval"
    assert logs[0]["context_length"] == 3
    assert logs[0]["query"] == query
    assert logs[0]["user_id"] == user_id

def test_canvas_logger_clear_logs():
    plugin = CanvasLoggerPlugin()
    plugin.before_retrieval("test", 1)
    assert len(plugin.get_logs()) == 1

    plugin.clear_logs()
    assert len(plugin.get_logs()) == 0
