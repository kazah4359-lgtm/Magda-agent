import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.llm_client import LLMClient
from magda_agent.architecture.magentic_one import MagenticOneOrchestrator

@pytest.mark.asyncio
async def test_magentic_one_orchestrator_success():
    mock_llm = AsyncMock(spec=LLMClient)

    # Mock behavior:
    # Evaluate difficulty: Local heuristic returns 2 (length < 20) -> team size 1
    # Plan call: returns JSON string (team size 1)
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
    # Evaluate difficulty: Local heuristic returns 2 (length < 20) -> team size 1
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
    # Evaluate difficulty: Local heuristic returns 2 (length < 20) -> team size 1
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

@pytest.mark.asyncio
async def test_magentic_one_orchestrator_dynamic_scaling():
    mock_llm = AsyncMock(spec=LLMClient)

    # Evaluate difficulty: length > 100 -> difficulty 10 -> team size 5
    mock_llm.chat_completion.side_effect = [
        '[{"id": "1", "description": "1"}, {"id": "2", "description": "2"}, {"id": "3", "description": "3"}, {"id": "4", "description": "4"}, {"id": "5", "description": "5"}]',
        "Done 1", "Done 2", "Done 3", "Done 4", "Done 5",
        "YES"
    ]

    orchestrator = MagenticOneOrchestrator(llm=mock_llm)
    # create a task string longer than 100 characters to trigger difficulty 10
    task_string = "Do the task " * 20
    await orchestrator.orchestrate(task_string)

    # 1 plan + 5 exec + 1 review = 7 calls
    assert mock_llm.chat_completion.call_count == 7
