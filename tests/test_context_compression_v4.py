"""
Unit tests for ClaudeSubagentContextCompressionV4 / ContextCompressionV4.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.architecture.context_compression_v4 import (
    ClaudeSubagentContextCompressionV4,
    ContextCompressionV4,
)


def test_class_alias():
    """Verify ContextCompressionV4 is an alias to ClaudeSubagentContextCompressionV4."""
    assert ContextCompressionV4 is ClaudeSubagentContextCompressionV4


def test_extract_critical_constraints():
    """Verify constraint sentence extraction."""
    compressor = ContextCompressionV4()
    text = (
        "Some random background info.\n"
        "You MUST never delete the system database.\n"
        "Normal turn statement.\n"
        "Strict rule: keep response under 100 words.\n"
    )
    constraints = compressor.extract_critical_constraints(text)
    assert len(constraints) == 2
    assert "You MUST never delete the system database." in constraints
    assert "Strict rule: keep response under 100 words." in constraints


def test_format_hierarchical_context():
    """Verify hierarchical context formatting with parent goal, subtask, and metadata."""
    compressor = ContextCompressionV4()
    history = [
        {"role": "user", "content": "How do I implement x?"},
        {"role": "assistant", "content": "You do it like this..."},
    ]
    metadata = {"worktree": "/tmp/worktree_1", "step_id": "step_3"}

    formatted = compressor.format_hierarchical_context(
        parent_goal="Build AGI Engine",
        subtask="Implement compression module",
        history=history,
        metadata=metadata,
    )

    assert len(formatted) == 3
    assert formatted[0]["role"] == "system"
    assert "Parent Goal: Build AGI Engine" in formatted[0]["content"]
    assert "Assigned Subtask: Implement compression module" in formatted[0]["content"]
    assert "worktree: /tmp/worktree_1" in formatted[0]["content"]
    assert formatted[1] == history[0]
    assert formatted[2] == history[1]


@pytest.mark.asyncio
async def test_compress_context_fallback_no_llm():
    """Verify fallback context compression when no LLM client is provided."""
    compressor = ContextCompressionV4(max_context_length=50)
    long_context = "This is a very long context string that exceeds the max length threshold.\nRule: MUST stay safe."

    compressed = await compressor.compress_context(long_context, task="Test subtask")
    assert len(compressed) <= 150  # Allows for constraint re-injection header
    assert "--- PRESERVED CRITICAL CONSTRAINTS ---" in compressed
    assert "Rule: MUST stay safe." in compressed


@pytest.mark.asyncio
async def test_compress_context_with_llm():
    """Verify LLM-driven context compression and constraint re-injection."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value="Condensed summary of context."
    )

    compressor = ContextCompressionV4(llm=mock_llm, max_context_length=20)
    context = (
        "This is an extended conversation history.\n"
        "Critical constraint: MUST preserve state in DB.\n"
    )

    result = await compressor.compress_context(context, task="Refactor DB")
    mock_llm.chat_completion.assert_called_once()
    assert "Condensed summary of context." in result
    assert "--- PRESERVED CRITICAL CONSTRAINTS ---" in result
    assert "Critical constraint: MUST preserve state in DB." in result


@pytest.mark.asyncio
async def test_compress_messages_with_llm():
    """Verify message history compression with mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value="Summary of historical discussion on refactoring."
    )

    compressor = ContextCompressionV4(llm=mock_llm, max_messages=2)
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Turn 3"},
        {"role": "assistant", "content": "Response 3"},
    ]

    compressed = await compressor.compress_messages(messages, task="Task A")
    # Expected: system message + 1 summary system message + last 2 user/assistant messages
    assert len(compressed) == 4
    assert compressed[0]["role"] == "system"
    assert compressed[1]["role"] == "system"
    assert "Compressed Context Summary:" in compressed[1]["content"]
    assert compressed[2] == messages[-2]
    assert compressed[3] == messages[-1]


@pytest.mark.asyncio
async def test_compress_payload():
    """Verify payload dictionary context and messages compression."""
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value="Compressed summary text."
    )

    compressor = ContextCompressionV4(llm=mock_llm, max_context_length=10, max_messages=2)
    payload = {
        "task": "Build feature",
        "context": "Very long context string exceeding 10 characters.",
        "messages": [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Turn 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Turn 4"},
        ]
    }

    result = await compressor.compress_payload(payload)
    assert result["task"] == "Build feature"
    assert "Compressed summary text." in result["context"]
    assert len(result["messages"]) == 3  # Summary message + 2 retained turns


@pytest.mark.asyncio
async def test_compress_hierarchical_context():
    """Verify compress_hierarchical_context convenience method."""
    compressor = ContextCompressionV4(max_messages=2)
    history = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Turn 2"},
        {"role": "user", "content": "Turn 3"},
        {"role": "assistant", "content": "Turn 4"},
    ]

    result = await compressor.compress_hierarchical_context(
        parent_goal="Deploy V4",
        subtask="Run test suite",
        history=history,
    )

    assert result[0]["role"] == "system"
    assert "Parent Goal: Deploy V4" in result[0]["content"]
    assert len(result) == 4  # System msg + Summary msg + 2 retained turns
