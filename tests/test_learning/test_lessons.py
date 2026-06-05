import pytest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.learning.lessons import TaskRecoveryLessons
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.llm_client import LLMClient

@pytest.fixture
def procedural_memory():
    # Use ephemeral client for testing to prevent persisting data to disk
    return ProceduralMemory(persist_directory=":memory:")

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock(return_value="Always ensure dependencies are installed before running tests.")
    return llm

@pytest.fixture
def task_recovery_lessons(procedural_memory, mock_llm):
    return TaskRecoveryLessons(procedural_memory=procedural_memory, llm=mock_llm)

@pytest.mark.asyncio
async def test_generate_and_store_lesson(task_recovery_lessons, mock_llm, procedural_memory):
    task_desc = "Run unit tests for the learning module"
    failure_reason = "ModuleNotFoundError: No module named pytest"
    user_id = 42

    await task_recovery_lessons.generate_and_store_lesson(task_desc, failure_reason, user_id=user_id)

    # Verify LLM was called to generate the lesson
    mock_llm.chat_completion.assert_called_once()

    # Retrieve to verify it was stored
    results = task_recovery_lessons.retrieve_relevant_lessons(task_desc, user_id=user_id)

    assert len(results) > 0
    # Our mock LLM returns "Always ensure dependencies are installed before running tests."
    assert "Always ensure dependencies are installed before running tests." in results[0]

@pytest.mark.asyncio
async def test_retrieve_relevant_lessons_no_results(task_recovery_lessons):
    # Should be empty for a new query
    results = task_recovery_lessons.retrieve_relevant_lessons("Some completely unrelated task", user_id=1)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_generate_empty_lesson(task_recovery_lessons, mock_llm, procedural_memory):
    # If the LLM returns an empty string, it shouldn't store anything
    mock_llm.chat_completion = AsyncMock(return_value="")

    # Use a unique user_id to ensure isolation from other tests
    await task_recovery_lessons.generate_and_store_lesson("unique_empty_task", "failure", user_id=999)

    results = task_recovery_lessons.retrieve_relevant_lessons("unique_empty_task", user_id=999)
    assert len(results) == 0
