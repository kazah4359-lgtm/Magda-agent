import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.skills.hermes_skills import HermesSkillCreator
from magda_agent.skills.registry import SkillRegistry

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_generate_skill(mock_llm_client):
    mock_llm_client.chat_completion.return_value = '''
```python
def test_skill(**kwargs):
    """
    A test skill
    """
    return "Skill executed: test_skill"
```
'''
    creator = HermesSkillCreator(llm_client=mock_llm_client)
    code = await creator.generate_skill("test_skill", "A test skill", "Do testing")

    assert "def test_skill(**kwargs):" in code
    assert "A test skill" in code
    assert "test_skill" in creator.created_skills

def test_load_skill_dynamically():
    creator = HermesSkillCreator()
    registry = SkillRegistry()

    code = '''
def test_skill(**kwargs):
    """
    A test skill
    """
    return "Skill executed: test_skill"
'''
    creator.created_skills["test_skill"] = code

    success = creator.load_skill_dynamically(registry, "test_skill")
    assert success is True

    # The skill should now be in the registry
    assert registry.has_skill("test_skill")

    # We should be able to execute it
    result = registry.execute_skill("test_skill")
    assert "Skill executed: test_skill" in result
