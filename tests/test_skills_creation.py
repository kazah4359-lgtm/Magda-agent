import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.learning.skills_creation import ExperienceSkillCreator
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client() -> LLMClient:
    client = MagicMock(spec=LLMClient)
    client.chat_completion = AsyncMock()
    return client

@pytest.fixture
def mock_procedural_memory() -> ProceduralMemory:
    mem = MagicMock(spec=ProceduralMemory)
    mem.store_procedure = MagicMock()
    return mem

@pytest.mark.asyncio
async def test_generate_skill_success(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
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

    creator = ExperienceSkillCreator(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)

    trace = [{"action": "read logs", "outcome": "got logs"}, {"action": "filter errors", "outcome": "got errors"}]
    result = await creator.generate_skill_from_experience("Parse error logs", trace)

    assert result is not None
    assert "def parse_logs" in result["code"]
    assert result["schema"]["name"] == "parse_logs"
    assert "parse_logs" in creator.created_skills

    mock_procedural_memory.store_procedure.assert_called_once_with(
        name="parse_logs",
        procedure="def parse_logs(logs):\n    return [l for l in logs if \"ERROR\" in l]",
        metadata={
            "source_task": "Parse error logs",
            "type": "hermes_experience_skill_v3",
            "schema": result["schema"]
        }
    )

@pytest.mark.asyncio
async def test_generate_skill_missing_code(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
    mock_response = """
    No code here
    ```json
    {
        "name": "parse_logs"
    }
    ```
    """
    mock_llm_client.chat_completion.return_value = mock_response

    creator = ExperienceSkillCreator(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_experience("Task", [{"action": "a", "outcome": "b"}])

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()

@pytest.mark.asyncio
async def test_generate_skill_invalid_json(mock_llm_client: LLMClient, mock_procedural_memory: ProceduralMemory) -> None:
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

    creator = ExperienceSkillCreator(llm_client=mock_llm_client, procedural_memory=mock_procedural_memory)
    result = await creator.generate_skill_from_experience("Task", [{"action": "a", "outcome": "b"}])

    assert result is None
    mock_procedural_memory.store_procedure.assert_not_called()
