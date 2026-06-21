import pytest
from unittest.mock import AsyncMock
from magda_agent.skills.creation import ExperienceSkillCreator

@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_generate_skill_success(mock_llm_client) -> None:
    mock_response = """
Here is the skill:
```python
def parse_logs(logs):
    return [l for l in logs if "ERROR" in l]
```
And the schema:
```json
{
    "name": "parse_logs",
    "description": "Parses error logs",
    "parameters": {
        "type": "object",
        "properties": {
            "logs": {"type": "array"}
        }
    }
}
```
"""
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreator(llm_client=mock_llm_client)

    trace = [{"action": "read logs", "outcome": "got logs"}, {"action": "filter errors", "outcome": "got errors"}]
    result = await creator.generate_skill_from_experience("Parse error logs", trace)

    assert result is not None
    assert "def parse_logs" in result["code"]
    assert result["schema"]["name"] == "parse_logs"
    assert "parse_logs" in creator.created_skills

@pytest.mark.asyncio
async def test_generate_skill_missing_code(mock_llm_client) -> None:
    mock_response = """
    No code here
    ```json
    {
        "name": "parse_logs"
    }
    ```
    """
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreator(llm_client=mock_llm_client)
    result = await creator.generate_skill_from_experience("Task", [{"action": "a", "outcome": "b"}])

    assert result is None

@pytest.mark.asyncio
async def test_generate_skill_invalid_json(mock_llm_client) -> None:
    mock_response = """
    ```python
    def foo(): pass
    ```
    ```json
    {
        "name": "foo",
        "invalid
    ```
    """
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreator(llm_client=mock_llm_client)
    result = await creator.generate_skill_from_experience("Task", [{"action": "a", "outcome": "b"}])

    assert result is None

@pytest.mark.asyncio
async def test_generate_skill_no_llm() -> None:
    creator = ExperienceSkillCreator(llm_client=None)
    with pytest.raises(ValueError):
        await creator.generate_skill_from_experience("Task", [])
