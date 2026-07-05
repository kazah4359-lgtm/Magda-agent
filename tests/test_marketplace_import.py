import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.marketplace_import import MarketplaceImporter

@pytest.fixture
def mock_registry() -> SkillRegistry:
    """Provides a fresh SkillRegistry instance."""
    return SkillRegistry()

@pytest.fixture
def importer(mock_registry: SkillRegistry) -> MarketplaceImporter:
    """Provides a MarketplaceImporter configured with a mock registry."""
    return MarketplaceImporter(registry=mock_registry)

@pytest.mark.asyncio
async def test_fetch_marketplace_catalog(importer: MarketplaceImporter) -> None:
    """Tests that fetching the catalog handles the aiohttp response correctly."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"skills": [{"name": "test_skill", "description": "test"}]}

    # Create a mock session context manager
    class MockClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def get(self, url):
            class MockGetContext:
                async def __aenter__(self):
                    return mock_response

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            return MockGetContext()

    with patch("aiohttp.ClientSession", return_value=MockClientSession()):
        catalog = await importer.fetch_marketplace_catalog()
        assert len(catalog) == 1
        assert catalog[0]["name"] == "test_skill"

@pytest.mark.asyncio
async def test_discover_skills(importer: MarketplaceImporter) -> None:
    """Tests filtering skills via a search query."""
    with patch.object(importer, "fetch_marketplace_catalog", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [
            {"name": "calculator", "description": "does math"},
            {"name": "weather", "description": "gets weather"}
        ]

        results = await importer.discover_skills("math")
        assert len(results) == 1
        assert results[0]["name"] == "calculator"

        all_results = await importer.discover_skills()
        assert len(all_results) == 2

@pytest.mark.asyncio
async def test_import_skill(importer: MarketplaceImporter, mock_registry: SkillRegistry) -> None:
    """Tests importing a specific skill definition into the registry."""
    skill_def = {
        "name": "test_import",
        "description": "test description",
        "parameters": {"type": "object", "properties": {}}
    }
    success = await importer.import_skill(skill_def)

    assert success is True
    assert "test_import" in mock_registry.skills

    # Test execution of imported skill
    skill_func = mock_registry.skills["test_import"]
    result = skill_func()
    assert "Executed remote skill" in result

@pytest.mark.asyncio
async def test_import_skill_by_name(importer: MarketplaceImporter, mock_registry: SkillRegistry) -> None:
    """Tests finding and importing a skill directly by its name."""
    with patch.object(importer, "fetch_marketplace_catalog", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [
            {"name": "target_skill", "description": "target"}
        ]

        success = await importer.import_skill_by_name("target_skill")
        assert success is True
        assert "target_skill" in mock_registry.skills

        success_not_found = await importer.import_skill_by_name("unknown")
        assert success_not_found is False
