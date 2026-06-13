import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.llm_client import LLMClient
from magda_agent.agents.magentic_one import MagenticOneOrchestrator

@pytest.mark.asyncio
async def test_magentic_one_orchestrator_success():
    mock_llm = AsyncMock(spec=LLMClient)

    # Mock behavior:
    # Plan call: returns JSON string
    # Execute call: returns "Subtask done"
    # Review call: returns "YES Result complete"
    mock_llm.chat_completion.side_effect = [
        '[{"id": "test_1", "description": "Execute first part of task"}]',
        "Subtask done",
        "YES Result complete"
    ]

    orchestrator = MagenticOneOrchestrator(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert result == "YES Result complete"
    assert mock_llm.chat_completion.call_count == 3

@pytest.mark.asyncio
async def test_magentic_one_orchestrator_max_iterations():
    mock_llm = AsyncMock(spec=LLMClient)

    # Mock behavior to always say NO, testing the iteration loop limit.
    # 3 iterations * 3 LLM calls each = 9 calls total.
    mock_llm.chat_completion.side_effect = [
        '[{"id": "test_1", "description": "Execute first part of task"}]', "Execute", "NO",
        '[{"id": "test_1", "description": "Execute first part of task"}]', "Execute", "NO",
        '[{"id": "test_1", "description": "Execute first part of task"}]', "Execute", "NO"
    ]

    orchestrator = MagenticOneOrchestrator(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task", max_iterations=3)

    assert "Task incomplete after 3 iterations" in result
    assert mock_llm.chat_completion.call_count == 9

@pytest.mark.asyncio
async def test_magentic_one_orchestrator_invalid_json():
    mock_llm = AsyncMock(spec=LLMClient)

    # Mock behavior:
    # Plan call: returns Invalid JSON string
    # Execute call: returns "Fallback task executed"
    # Review call: returns "YES Result complete"
    mock_llm.chat_completion.side_effect = [
        'Invalid JSON',
        "Fallback task executed",
        "YES Result complete"
    ]

    orchestrator = MagenticOneOrchestrator(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert result == "YES Result complete"
    assert mock_llm.chat_completion.call_count == 3
