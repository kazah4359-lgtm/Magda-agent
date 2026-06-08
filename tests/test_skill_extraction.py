import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.skill_extraction import SkillExtractor
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock()
    return llm

@pytest.fixture
def procedural_memory():
    return ProceduralMemory(persist_directory=":memory:")

@pytest.fixture
def skill_extractor(procedural_memory, mock_llm):
    return SkillExtractor(procedural_memory=procedural_memory, llm=mock_llm)

@pytest.mark.asyncio
async def test_extract_skill_success(skill_extractor, mock_llm, procedural_memory):
    mock_llm.chat_completion.return_value = "Extracted reusable skill procedure."

    task_description = "Process data"
    experience_logs = [
        {"action": "Read file", "outcome": "Success"},
        {"action": "Parse JSON", "outcome": "Success"}
    ]
    user_id = 123

    await skill_extractor.extract_skill(task_description, experience_logs, user_id=user_id)

    mock_llm.chat_completion.assert_called_once()
    results = procedural_memory.recall_procedure(query="Process data", user_id=user_id)

    assert len(results) > 0
    assert "Extracted reusable skill procedure." in results[0]

@pytest.mark.asyncio
async def test_extract_skill_empty(skill_extractor, mock_llm, procedural_memory):
    mock_llm.chat_completion.return_value = "   "

    task_description = "Empty extraction"
    experience_logs = [
        {"action": "Do nothing", "outcome": "Success"}
    ]
    user_id = 999

    await skill_extractor.extract_skill(task_description, experience_logs, user_id=user_id)

    mock_llm.chat_completion.assert_called_once()
    results = procedural_memory.recall_procedure(query="Empty", user_id=user_id)
    assert len(results) == 0
