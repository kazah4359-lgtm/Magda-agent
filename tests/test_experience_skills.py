import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.experience_skills import HermesSkillExperienceManager


@pytest.mark.asyncio
async def test_create_skill_from_experience_success() -> None:
    """Tests that a valid Python function is parsed, stored, and returned."""
    mock_llm = AsyncMock()
    mock_memory = MagicMock()

    manager = HermesSkillExperienceManager(llm=mock_llm, procedural_memory=mock_memory)

    mock_code = "def sample_experience_skill(data):\n    return data.get('key')"
    mock_llm.chat_completion.return_value = mock_code

    trace = [
        {"action": "read data", "outcome": "success"},
        {"action": "extract key", "outcome": "success"}
    ]

    result = await manager.create_skill_from_experience("Extract key from data dict", trace, user_id=42)

    assert result == mock_code

    mock_memory.store_procedure.assert_called_once()
    args, kwargs = mock_memory.store_procedure.call_args

    assert kwargs["name"] == "sample_experience_skill"
    assert kwargs["procedure"] == mock_code
    assert kwargs["user_id"] == 42
    assert kwargs["metadata"]["source_task"] == "Extract key from data dict"
    assert kwargs["metadata"]["type"] == "hermes_experience_skill"


@pytest.mark.asyncio
async def test_create_skill_from_experience_not_reusable() -> None:
    """Tests that a non-reusable trace does not trigger storage and returns None."""
    mock_llm = AsyncMock()
    mock_memory = MagicMock()

    manager = HermesSkillExperienceManager(llm=mock_llm, procedural_memory=mock_memory)

    mock_llm.chat_completion.return_value = "NOT_REUSABLE"

    trace = [{"action": "one-off API call", "outcome": "success"}]

    result = await manager.create_skill_from_experience("One-off task", trace)

    assert result is None
    mock_memory.store_procedure.assert_not_called()


@pytest.mark.asyncio
async def test_create_skill_from_experience_empty_trace() -> None:
    """Tests that an empty trace returns None without calling LLM."""
    mock_llm = AsyncMock()
    mock_memory = MagicMock()

    manager = HermesSkillExperienceManager(llm=mock_llm, procedural_memory=mock_memory)

    result = await manager.create_skill_from_experience("Empty trace task", [])

    assert result is None
    mock_llm.chat_completion.assert_not_called()
    mock_memory.store_procedure.assert_not_called()
