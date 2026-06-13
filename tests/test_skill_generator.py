import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.skills.skill_generator import SkillGenerator

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """
    Provides a mocked LLMClient.
    """
    client = MagicMock()
    client.chat_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_generate_skill_from_queries(mock_llm_client: MagicMock) -> None:
    """
    Tests that a skill is generated from user queries.
    """
    mock_llm_client.chat_completion.return_value = '''```python
def synthesized_skill(**kwargs):
    return "This is a synthesized skill."
```'''
    generator = SkillGenerator(llm_client=mock_llm_client)
    queries = ["What time is it?", "Tell me the time", "Current time"]
    code = await generator.generate_skill_from_queries(queries)

    assert code is not None
    assert "def synthesized_skill(**kwargs):" in code
    assert "synthesized_skill" in generator.synthesized_skills

@pytest.mark.asyncio
async def test_generate_skill_sandbox_violation(mock_llm_client: MagicMock) -> None:
    """
    Tests that a skill violating sandbox policies is rejected.
    """
    mock_llm_client.chat_completion.return_value = '''```python
import os
def synthesized_skill(**kwargs):
    return os.getenv("PATH")
```'''
    generator = SkillGenerator(llm_client=mock_llm_client)
    queries = ["Show environment variables"]
    code = await generator.generate_skill_from_queries(queries)

    assert code is None
    assert "synthesized_skill" not in generator.synthesized_skills
