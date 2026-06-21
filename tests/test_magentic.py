import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.agents.magentic import Worker, Director
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_worker_execution_with_llm():
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_completion.return_value = "Worker result"

    worker = Worker("Coder", "Writes code", llm=mock_llm)
    result = await worker.execute_task("Write a function")

    assert result == "Worker result"
    mock_llm.chat_completion.assert_called_once()
    args, kwargs = mock_llm.chat_completion.call_args
    assert "You are Coder" in args[0][0]["content"]

@pytest.mark.asyncio
async def test_worker_execution_no_llm():
    worker = Worker("MockWorker", "Mocks things", llm=None)
    result = await worker.execute_task("Do a mock task")

    assert "executed task: Do a mock task" in result

@pytest.mark.asyncio
async def test_director_delegation():
    mock_llm = AsyncMock(spec=LLMClient)

    # Mock sequence:
    # 1. Assignment JSON
    # 2. Final synthesis
    mock_llm.chat_completion.side_effect = [
        '[{"worker_name": "Coder", "subtask": "write code"}, {"worker_name": "Reviewer", "subtask": "review code"}]',
        "Final synthesis result"
    ]

    mock_worker_1 = AsyncMock(spec=Worker)
    mock_worker_1.name = "Coder"
    mock_worker_1.description = "Writes code"
    mock_worker_1.execute_task.return_value = "Code written"

    mock_worker_2 = AsyncMock(spec=Worker)
    mock_worker_2.name = "Reviewer"
    mock_worker_2.description = "Reviews code"
    mock_worker_2.execute_task.return_value = "Code reviewed"

    director = Director(llm=mock_llm, workers=[mock_worker_1, mock_worker_2])

    result = await director.delegate("Write and review code")

    assert result == "Final synthesis result"
    assert mock_llm.chat_completion.call_count == 2

    # Verify workers were called
    mock_worker_1.execute_task.assert_called_once_with("write code")
    mock_worker_2.execute_task.assert_called_once_with("review code")

@pytest.mark.asyncio
async def test_director_delegation_parsing_error():
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_completion.return_value = "I am not returning JSON today."

    director = Director(llm=mock_llm, workers=[])
    result = await director.delegate("Task")

    assert result == "Delegation failed: Could not create assignments."
