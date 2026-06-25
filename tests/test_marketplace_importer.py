import pytest
import json
import os
import shutil
from unittest.mock import patch, mock_open, MagicMock, AsyncMock
from aiohttp import ClientError
from magda_agent.skills.marketplace_importer import SkillMarketplaceImporter
from magda_agent.skills.marketplace import load_skill_from_marketplace
from magda_agent.skills.registry import SkillRegistry

@pytest.fixture
def temp_cache_dir(tmp_path) -> str:
    """Fixture for creating a temporary directory for skill caches."""
    cache_dir = tmp_path / ".test_skill_cache"
    yield str(cache_dir)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

@pytest.fixture
def importer(temp_cache_dir) -> SkillMarketplaceImporter:
    """Fixture to provide a SkillMarketplaceImporter instance."""
    return SkillMarketplaceImporter(cache_dir=temp_cache_dir)

def test_initialization(temp_cache_dir) -> None:
    """Tests the initialization of the importer and cache directory creation."""
    imp = SkillMarketplaceImporter(cache_dir=temp_cache_dir)
    assert os.path.exists(temp_cache_dir)
    assert imp.memory_cache == {}

@pytest.mark.asyncio
async def test_get_skill_from_memory_cache(importer) -> None:
    """Tests that a skill is correctly returned from the memory cache."""
    mock_skill = {"name": "test_skill", "description": "A test skill"}
    importer.memory_cache["test_skill"] = mock_skill

    skill = await importer.get_skill("test_skill")
    assert skill == mock_skill

@pytest.mark.asyncio
async def test_get_skill_from_disk_cache(importer, temp_cache_dir) -> None:
    """Tests that a skill is returned from the disk cache when not in memory."""
    mock_skill = {"name": "test_skill", "description": "A test skill from disk"}
    skill_path = os.path.join(temp_cache_dir, "test_skill.json")

    with open(skill_path, 'w', encoding='utf-8') as f:
        json.dump(mock_skill, f)

    skill = await importer.get_skill("test_skill")
    assert skill == mock_skill
    assert importer.memory_cache["test_skill"] == mock_skill

@pytest.mark.asyncio
@patch('magda_agent.skills.marketplace_importer.aiohttp.ClientSession.get')
async def test_get_skill_from_network_success(mock_get, importer, temp_cache_dir) -> None:
    """Tests a successful fetch of a skill from the network."""
    mock_skill = {"name": "network_skill", "description": "A test skill from network"}
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_skill
    mock_response.__aenter__.return_value = mock_response
    mock_get.return_value = mock_response

    skill = await importer.get_skill("network_skill")

    assert skill == mock_skill
    assert importer.memory_cache["network_skill"] == mock_skill

    skill_path = os.path.join(temp_cache_dir, "network_skill.json")
    assert os.path.exists(skill_path)
    with open(skill_path, 'r', encoding='utf-8') as f:
        saved_skill = json.load(f)
    assert saved_skill == mock_skill

@pytest.mark.asyncio
@patch('magda_agent.skills.marketplace_importer.aiohttp.ClientSession.get')
async def test_get_skill_from_network_failure(mock_get, importer) -> None:
    """Tests fetching a skill from the network when a network error occurs."""
    mock_get.side_effect = ClientError("Network error")

    skill = await importer.get_skill("failed_skill")
    assert skill is None
    assert "failed_skill" not in importer.memory_cache

@pytest.mark.asyncio
async def test_invalidate_specific_skill_cache(importer, temp_cache_dir) -> None:
    """Tests cache invalidation for a specific skill."""
    mock_skill = {"name": "test_skill"}
    importer.memory_cache["test_skill"] = mock_skill
    skill_path = os.path.join(temp_cache_dir, "test_skill.json")
    with open(skill_path, 'w', encoding='utf-8') as f:
        json.dump(mock_skill, f)

    await importer.invalidate_cache("test_skill")

    assert "test_skill" not in importer.memory_cache
    assert not os.path.exists(skill_path)

@pytest.mark.asyncio
async def test_invalidate_all_cache(importer, temp_cache_dir) -> None:
    """Tests total invalidation of the skill cache."""
    mock_skill = {"name": "test_skill"}
    importer.memory_cache["test_skill"] = mock_skill
    skill_path = os.path.join(temp_cache_dir, "test_skill.json")
    with open(skill_path, 'w', encoding='utf-8') as f:
        json.dump(mock_skill, f)

    await importer.invalidate_cache()

    assert importer.memory_cache == {}
    assert os.path.exists(temp_cache_dir)
    assert not os.path.exists(skill_path)

@pytest.mark.asyncio
async def test_path_traversal_prevention(importer, temp_cache_dir) -> None:
    """Tests that a malicious skill name cannot lead to a path traversal attack."""
    malicious_name = "../../../malicious_skill"
    cache_path = importer._get_cache_path(malicious_name)
    assert temp_cache_dir in cache_path

    # We URL encode the input, so "../" becomes "..%2F"
    # To check path traversal prevention we just ensure it's a valid relative path within cache_dir
    assert os.path.relpath(cache_path, temp_cache_dir).startswith("..%2F") or ".." not in os.path.relpath(cache_path, temp_cache_dir).split(os.sep)

@pytest.mark.asyncio
@patch('magda_agent.skills.marketplace_importer.aiohttp.ClientSession.get')
async def test_load_skill_from_marketplace(mock_get, importer) -> None:
    """Tests integrating with the marketplace loader by loading and registering a skill."""
    mock_skill = {"name": "integration_skill", "description": "integration test"}
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_skill
    mock_response.__aenter__.return_value = mock_response
    mock_get.return_value = mock_response

    registry = SkillRegistry()
    success = await load_skill_from_marketplace("integration_skill", registry, importer)

    assert success is True
    assert "integration_skill" in registry.skills
    assert registry.descriptions["integration_skill"] == "integration test"
