import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.skill_creator import SkillCreator
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock()
    return llm

@pytest.fixture
def procedural_memory():
    # Use ephemeral client for testing to prevent persisting data to disk
    return ProceduralMemory(persist_directory=":memory:")

@pytest.fixture
def skill_creator(procedural_memory, mock_llm):
    return SkillCreator(procedural_memory=procedural_memory, llm=mock_llm)

@pytest.mark.asyncio
async def test_extract_and_store_skill_success(skill_creator, mock_llm, procedural_memory):
    mock_llm.chat_completion.return_value = "Always check for None before processing."

    task_description = "Handle empty input gracefully"
    execution_steps = [
        {"description": "Check input", "skill": "input_validator", "result": "Input is None"},
        {"description": "Return error", "skill": "error_handler", "result": "Returned error code 400"}
    ]
    user_id = 123

    await skill_creator.extract_and_store_skill(task_description, execution_steps, user_id=user_id)

    # Verify LLM was called
    mock_llm.chat_completion.assert_called_once()
    call_args = mock_llm.chat_completion.call_args[0][0][0]["content"]
    assert "Handle empty input gracefully" in call_args
    assert "input_validator" in call_args
    assert "error_handler" in call_args

    # Retrieve to verify it was stored in procedural memory
    results = procedural_memory.recall_procedure(query="Handle empty input", user_id=user_id)

    assert len(results) > 0
    assert "Always check for None before processing." in results[0]

@pytest.mark.asyncio
async def test_extract_and_store_skill_empty_llm_response(skill_creator, mock_llm, procedural_memory):
    mock_llm.chat_completion.return_value = "   " # Empty response

    task_description = "A task that shouldn't store anything"
    execution_steps = [
        {"description": "Step 1", "skill": "skill_1", "result": "Result 1"}
    ]

    # Use a unique user_id to avoid mixing with other tests
    user_id = 999

    await skill_creator.extract_and_store_skill(task_description, execution_steps, user_id=user_id)

    # Verify LLM was called
    mock_llm.chat_completion.assert_called_once()

    # Retrieve to verify NOTHING was stored
    results = procedural_memory.recall_procedure(query="A task", user_id=user_id)
    assert len(results) == 0
