"""
Tests for the SubagentSpawner module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.architecture.subagent_spawning import SubagentSpawner

def test_compress_context_short():
    """Test compressing a context that is already short enough."""
    spawner = SubagentSpawner()
    context = [
        {"role": "system", "content": "You are an AI."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    compressed = spawner.compress_context(context)
    assert len(compressed) == 3
    assert compressed == context

def test_compress_context_long():
    """Test compressing a context that is longer than 5 messages."""
    spawner = SubagentSpawner()
    context = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Msg 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Msg 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Msg 3"},
        {"role": "assistant", "content": "Reply 3"},
    ]
    compressed = spawner.compress_context(context)
    # Should keep first and last 4
    assert len(compressed) == 5
    assert compressed[0] == context[0]
    assert compressed[1:] == context[-4:]

def test_compress_context_empty():
    """Test compressing an empty context."""
    spawner = SubagentSpawner()
    compressed = spawner.compress_context([])
    assert compressed == []

@pytest.mark.asyncio
async def test_spawn_subagent_callable():
    """Test spawning a subagent using a callable executor."""
    spawner = SubagentSpawner()
    context = [{"role": "system", "content": "System"}]

    async def mock_executor(ctx):
        mock_executor.called = True
        mock_executor.call_args = ctx
        return "Task Complete"

    mock_executor.called = False

    result = await spawner.spawn_subagent("Do something", context, mock_executor)

    assert result == "Task Complete"
    assert mock_executor.called

    # Check the execution context passed
    called_context = mock_executor.call_args
    assert len(called_context) == 2
    assert called_context[0] == context[0]
    assert called_context[1]["role"] == "user"
    assert "Task: Do something" in called_context[1]["content"]

@pytest.mark.asyncio
async def test_spawn_subagent_with_execute_method():
    """Test spawning a subagent using an executor with an execute method."""
    spawner = SubagentSpawner()
    context = [{"role": "system", "content": "System"}]

    executor = MagicMock()
    executor.execute = AsyncMock(return_value="Task Executed")

    result = await spawner.spawn_subagent("Do something else", context, executor)

    assert result == "Task Executed"
    executor.execute.assert_called_once()

@pytest.mark.asyncio
async def test_spawn_subagent_invalid_executor():
    """Test spawning a subagent with an invalid executor raises TypeError."""
    spawner = SubagentSpawner()
    context = []

    # Invalid executor (neither callable nor has execute method)
    invalid_executor = "not a callable"

    with pytest.raises(TypeError):
        await spawner.spawn_subagent("Fail task", context, invalid_executor)
