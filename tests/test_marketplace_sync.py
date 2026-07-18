import pytest
import respx
import httpx
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.marketplace_sync import MarketplaceSyncRoutine

@pytest.fixture
def registry():
    return SkillRegistry()

@pytest.fixture
def sync_routine(registry):
    return MarketplaceSyncRoutine(registry=registry)

@pytest.mark.asyncio
async def test_run_sync_cycle_success_list(sync_routine, registry):
    mock_data = [
        {
            "name": "test_skill_1",
            "description": "A test skill 1",
            "parameters": {}
        },
        {
            "name": "test_skill_2",
            "description": "A test skill 2",
            "inputSchema": {}
        }
    ]

    with respx.mock:
        respx.get("https://agentskills.io/api/skills").mock(return_value=httpx.Response(200, json=mock_data))

        imported_count = await sync_routine.run_sync_cycle()

        assert imported_count == 2
        assert registry.has_skill("test_skill_1")
        assert registry.has_skill("test_skill_2")

@pytest.mark.asyncio
async def test_run_sync_cycle_success_dict(sync_routine, registry):
    mock_data = {
        "skills": [
            {
                "name": "test_skill_3",
                "description": "A test skill 3",
                "parameters": {}
            }
        ]
    }

    with respx.mock:
        respx.get("https://agentskills.io/api/skills").mock(return_value=httpx.Response(200, json=mock_data))

        imported_count = await sync_routine.run_sync_cycle()

        assert imported_count == 1
        assert registry.has_skill("test_skill_3")

@pytest.mark.asyncio
async def test_run_sync_cycle_invalid_format(sync_routine, registry):
    mock_data = {"invalid": "data"}

    with respx.mock:
        respx.get("https://agentskills.io/api/skills").mock(return_value=httpx.Response(200, json=mock_data))

        imported_count = await sync_routine.run_sync_cycle()

        assert imported_count == 0
        assert registry.get_skills_summary().strip() == "Available Skills:"

@pytest.mark.asyncio
async def test_run_sync_cycle_http_error(sync_routine, registry):
    with respx.mock:
        respx.get("https://agentskills.io/api/skills").mock(return_value=httpx.Response(500))

        imported_count = await sync_routine.run_sync_cycle()

        assert imported_count == 0
        assert registry.get_skills_summary().strip() == "Available Skills:"
