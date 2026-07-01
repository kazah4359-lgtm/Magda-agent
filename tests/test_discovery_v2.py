import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.skills.discovery_v2 import SkillDiscoveryPipeline
import json

@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_completion = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_discover_skills_success(mock_llm_client):
    mock_llm_client.chat_completion.return_value = '''
```json
[
  {
    "name": "read_logs",
    "description": "Reads error logs from a specified file",
    "instructions": "Open the file and return the last 100 lines."
  },
  {
    "name": "restart_service",
    "description": "Restarts a system service",
    "instructions": "Run systemctl restart <service_name>."
  }
]
```
'''
    pipeline = SkillDiscoveryPipeline(llm_client=mock_llm_client)
    skills = await pipeline.discover_skills("I keep having to manually restart the web service after reading the error logs.")

    assert len(skills) == 2
    assert skills[0]["name"] == "read_logs"
    assert skills[1]["name"] == "restart_service"

@pytest.mark.asyncio
async def test_discover_skills_no_client():
    pipeline = SkillDiscoveryPipeline(llm_client=None)
    skills = await pipeline.discover_skills("Some text")
    assert skills == []

@pytest.mark.asyncio
async def test_discover_skills_llm_error(mock_llm_client):
    mock_llm_client.chat_completion.side_effect = Exception("LLM connection failed")
    pipeline = SkillDiscoveryPipeline(llm_client=mock_llm_client)
    skills = await pipeline.discover_skills("Some text")
    assert skills == []

@pytest.mark.asyncio
async def test_discover_skills_invalid_json(mock_llm_client):
    mock_llm_client.chat_completion.return_value = "This is not json."
    pipeline = SkillDiscoveryPipeline(llm_client=mock_llm_client)
    skills = await pipeline.discover_skills("Some text")
    assert skills == []

@pytest.mark.asyncio
async def test_discover_skills_json_not_list(mock_llm_client):
    mock_llm_client.chat_completion.return_value = '''
```json
{"name": "test", "description": "test", "instructions": "test"}
```
'''
    pipeline = SkillDiscoveryPipeline(llm_client=mock_llm_client)
    skills = await pipeline.discover_skills("Some text")
    assert skills == []

@pytest.mark.asyncio
async def test_discover_skills_json_missing_fields(mock_llm_client):
    mock_llm_client.chat_completion.return_value = '''
```json
[
  {"name": "valid", "description": "d", "instructions": "i"},
  {"name": "invalid", "description": "d"}
]
```
'''
    pipeline = SkillDiscoveryPipeline(llm_client=mock_llm_client)
    skills = await pipeline.discover_skills("Some text")

    # Should only return the valid skill
    assert len(skills) == 1
    assert skills[0]["name"] == "valid"
