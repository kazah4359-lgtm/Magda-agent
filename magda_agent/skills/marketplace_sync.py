import logging
import httpx
from typing import Optional, Any, Dict, List
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.agentskills_importer import AgentSkillsImporter

logger = logging.getLogger(__name__)

class MarketplaceSyncRoutine:
    """
    Routine for fetching, parsing, and synchronizing skills from an external marketplace.
    """

    def __init__(self, registry: SkillRegistry, marketplace_url: str = "https://agentskills.io/api/skills"):
        """
        Initializes the MarketplaceSyncRoutine.

        Args:
            registry: The SkillRegistry to sync skills into.
            marketplace_url: The URL of the external marketplace to fetch from.
        """
        self.registry = registry
        self.marketplace_url = marketplace_url
        self.importer = AgentSkillsImporter(registry=self.registry)

    async def fetch_marketplace_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches the skills data from the marketplace URL.

        Returns:
            The parsed JSON data (expected as a list of skills) or None if fetching fails.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.marketplace_url)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "skills" in data:
                    return data["skills"]
                else:
                    logger.error("Marketplace data format not recognized.")
                    return None
        except Exception as e:
            logger.error(f"Failed to fetch marketplace data from {self.marketplace_url}: {e}")
            return None

    async def run_sync_cycle(self) -> int:
        """
        Runs a complete sync cycle: fetching data and importing skills.

        Returns:
            The number of skills successfully imported.
        """
        logger.info(f"Starting marketplace sync cycle from {self.marketplace_url}")

        data = await self.fetch_marketplace_data()
        if not data:
            logger.warning("No data retrieved during marketplace sync.")
            return 0

        imported_funcs = self.importer.import_skills(data)
        logger.info(f"Marketplace sync complete. Successfully imported {len(imported_funcs)} skills.")
        return len(imported_funcs)
