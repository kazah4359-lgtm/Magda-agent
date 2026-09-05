import pytest
import json
from unittest.mock import AsyncMock
from magda_agent.llm_client import LLMClient
from magda_agent.architecture.magentic_one_v3 import MagenticOneOrchestratorV3, MagenticOneWorkerV3

@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_success():
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

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert result == "YES Result complete"
    assert mock_llm.chat_completion.call_count == 3

@pytest.mark.asyncio
async def test_magentic_one_v3_round_robin_execution():
    mock_llm = AsyncMock(spec=LLMClient)

    # Task > 100 characters gives difficulty 10 -> team size 5
    # Generate 5 tasks without explicit worker assignment
    plan_json = json.dumps([
        {"id": "1", "description": "Task 1"},
        {"id": "2", "description": "Task 2"},
        {"id": "3", "description": "Task 3"},
        {"id": "4", "description": "Task 4"},
        {"id": "5", "description": "Task 5"}
    ])

    mock_llm.chat_completion.side_effect = [
        plan_json,
        "Worker 1 done", # Exec 1 -> WebSurfer
        "Worker 2 done", # Exec 2 -> FileSurfer
        "Worker 3 done", # Exec 3 -> Coder
        "Worker 4 done", # Exec 4 -> Executor
        "Worker 5 done", # Exec 5 -> WebSurfer (round robin cycle)
        "YES Complete"   # Review
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    # Ensure starting from 0
    assert orchestrator._current_worker_index == 0

    # Task > 100 characters
    task_string = "Execute tasks " * 20
    await orchestrator.orchestrate(task_string)

    # 1 plan + 5 exec + 1 review = 7 calls
    assert mock_llm.chat_completion.call_count == 7

    # The current worker index should now be 1 because 5 % 4 = 1
    assert orchestrator._current_worker_index == 1

    # Let's inspect the execution calls to ensure they hit the right workers based on their description
    exec_calls = mock_llm.chat_completion.call_args_list[1:6]
    assert "WebSurfer" in exec_calls[0][0][0][0]["content"]
    assert "FileSurfer" in exec_calls[1][0][0][0]["content"]
    assert "Coder" in exec_calls[2][0][0][0]["content"]
    assert "Executor" in exec_calls[3][0][0][0]["content"]
    assert "WebSurfer" in exec_calls[4][0][0][0]["content"] # Cycle back


@pytest.mark.asyncio
async def test_magentic_one_v3_explicit_worker():
    mock_llm = AsyncMock(spec=LLMClient)

    # Evaluate difficulty: Local heuristic returns 2 -> team size 1
    plan_json = json.dumps([
        {"id": "test_1", "description": "Execute first part of task", "worker": "Coder"}
    ])

    mock_llm.chat_completion.side_effect = [
        plan_json,
        "Coder task done",
        "YES Result complete"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert result == "YES Result complete"
    assert mock_llm.chat_completion.call_count == 3

    exec_call = mock_llm.chat_completion.call_args_list[1]
    assert "Coder" in exec_call[0][0][0]["content"]
    # Ensure round-robin index hasn't advanced because an explicit worker was specified
    assert orchestrator._current_worker_index == 0

@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_max_iterations():
    mock_llm = AsyncMock(spec=LLMClient)

    # 3 iterations * 3 LLM calls each = 9 calls total.
    mock_llm.chat_completion.side_effect = [
        '[{"id": "test_1", "description": "Execute"}]', "Execute", "NO",
        '[{"id": "test_1", "description": "Execute"}]', "Execute", "NO",
        '[{"id": "test_1", "description": "Execute"}]', "Execute", "NO"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task", max_iterations=3)

    assert "Task incomplete after 3 iterations" in result
    assert mock_llm.chat_completion.call_count == 9

@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_invalid_json():
    mock_llm = AsyncMock(spec=LLMClient)

    mock_llm.chat_completion.side_effect = [
        'Invalid JSON',
        "Fallback task executed",
        "YES Result complete"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Do the task")

    assert result == "YES Result complete"
    assert mock_llm.chat_completion.call_count == 3


@pytest.mark.asyncio
async def test_magentic_one_v3_orchestrator_hierarchical_delegation():
    mock_llm = AsyncMock(spec=LLMClient)

    mock_llm.chat_completion.side_effect = [
        '[{"id": "parent_1", "description": "Parent task", "subtasks": [{"id": "child_1", "description": "Child task 1"}, {"id": "child_2", "description": "Child task 2"}]}]',
        "Child 1 done",
        "Child 2 done",
        "YES Complete"
    ]

    orchestrator = MagenticOneOrchestratorV3(llm=mock_llm)
    result = await orchestrator.orchestrate("Complex hierarchical task")

    assert result == "YES Complete"
    # Plan + Child 1 + Child 2 + Review = 4 calls
    assert mock_llm.chat_completion.call_count == 4

    # Because subtasks don't define worker, they go to WebSurfer then FileSurfer
    exec_call_1 = mock_llm.chat_completion.call_args_list[1]
    exec_call_2 = mock_llm.chat_completion.call_args_list[2]

    assert "WebSurfer" in exec_call_1[0][0][0]["content"]
    assert "FileSurfer" in exec_call_2[0][0][0]["content"]
    assert orchestrator._current_worker_index == 2
