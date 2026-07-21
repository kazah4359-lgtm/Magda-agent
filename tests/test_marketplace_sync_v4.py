import pytest
import respx
import httpx
import json
import os
import asyncio
from unittest.mock import patch, MagicMock
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.marketplace_sync_v4 import MarketplaceSyncRoutineV4

@pytest.fixture
def registry():
    return SkillRegistry()

@pytest.fixture
def cache_file(tmp_path):
    return os.path.join(tmp_path, "test_cache_v4.json")

@pytest.fixture
def sync_routine(registry, cache_file):
    return MarketplaceSyncRoutineV4(
        registry=registry,
        marketplace_url="https://agentskills.io/api/skills",
        interval=0.05,
        cache_path=cache_file
    )

@pytest.mark.asyncio
async def test_run_sync_cycle_success(sync_routine, registry, cache_file):
    mock_data = [
        {
            "name": "v4_test_skill_1",
            "description": "V4 Test Skill 1 description",
            "parameters": {}
        },
        {
            "name": "v4_test_skill_2",
            "description": "V4 Test Skill 2 description",
            "inputSchema": {}
        }
    ]

    with respx.mock:
        respx.get("https://agentskills.io/api/skills").mock(return_value=httpx.Response(200, json=mock_data))

        imported_count = await sync_routine.run_sync_cycle()

        assert imported_count == 2
        assert registry.has_skill("v4_test_skill_1")
        assert registry.has_skill("v4_test_skill_2")

        # Verify cached file is written
        assert os.path.exists(cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_json = json.load(f)
        assert cached_json == mock_data

@pytest.mark.asyncio
async def test_run_sync_cycle_fallback_to_cache(sync_routine, registry, cache_file):
    # Pre-populate cache
    cached_data = [
        {
            "name": "v4_cached_skill",
            "description": "This comes from local cache"
        }
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_data, f)

    with respx.mock:
        # Mocking API error
        respx.get("https://agentskills.io/api/skills").mock(return_value=httpx.Response(500))

        imported_count = await sync_routine.run_sync_cycle()

        # Should fallback to cache and import 1 skill
        assert imported_count == 1
        assert registry.has_skill("v4_cached_skill")

@pytest.mark.asyncio
async def test_periodic_fetching_loop(sync_routine, registry):
    mock_data = [
        {
            "name": "v4_periodic_skill",
            "description": "Periodic fetch skill"
        }
    ]

    with respx.mock:
        route = respx.get("https://agentskills.io/api/skills").mock(return_value=httpx.Response(200, json=mock_data))

        # Start periodic sync with fast interval
        await sync_routine.start()
        assert sync_routine._running is True

        # Let the loop run for a short duration to execute a couple of syncs
        await asyncio.sleep(0.18)

        await sync_routine.stop()
        assert sync_routine._running is False
        assert sync_routine._task is None

        # Verify that multiple calls were made to the mock API
        assert route.called
        assert len(respx.calls) >= 2
        assert registry.has_skill("v4_periodic_skill")
