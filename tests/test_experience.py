import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.experience import ExperienceToSkillCreator

@pytest.mark.asyncio
async def test_create_skill_from_experience_success() -> None:
    mock_memory = MagicMock()
    mock_llm = AsyncMock()

    mock_code = "def sample_skill(x):\n    return x + 1"
    mock_llm.chat_completion.return_value = mock_code

    creator = ExperienceToSkillCreator(procedural_memory=mock_memory, llm=mock_llm)

    trace = [{"action": "add 1", "outcome": "success"}]

    result = await creator.create_skill_from_experience("add one to number", trace)

    assert result == mock_code
    mock_memory.store_procedure.assert_called_once()
    args, kwargs = mock_memory.store_procedure.call_args
    assert kwargs["name"] == "sample_skill"
    assert kwargs["procedure"] == mock_code

@pytest.mark.asyncio
async def test_create_skill_from_experience_not_reusable() -> None:
    mock_memory = MagicMock()
    mock_llm = AsyncMock()

    mock_llm.chat_completion.return_value = "NOT_REUSABLE"

    creator = ExperienceToSkillCreator(procedural_memory=mock_memory, llm=mock_llm)

    trace = [{"action": "random action", "outcome": "failure"}]

    result = await creator.create_skill_from_experience("random task", trace)

    assert result is None
    mock_memory.store_procedure.assert_not_called()
