import json
import logging
import inspect
from typing import Dict, Any, List, Optional
import aiohttp

from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.marketplace import _create_dynamic_skill

logger = logging.getLogger(__name__)

class MarketplaceImporter:
    """
    Interface for discovering and importing skills from agentskills.io format.
    """
    def __init__(self, registry: SkillRegistry, base_url: str = "https://registry.agentskills.io/api/v1/skills") -> None:
        """
        Initializes the MarketplaceImporter.
        """
        self.registry = registry
        self.base_url = base_url

    async def fetch_marketplace_catalog(self) -> List[Dict[str, Any]]:
        """
        Fetches the complete catalog of skills from the marketplace.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    response.raise_for_status()
                    data = await response.json()
                    # Assume data contains a 'skills' key or is a list itself
                    if isinstance(data, dict):
                        return data.get("skills", [])
                    elif isinstance(data, list):
                        return data
                    return []
        except Exception as e:
            logger.error(f"Failed to fetch catalog from {self.base_url}: {e}")
            return []

    async def discover_skills(self, query: str = "") -> List[Dict[str, Any]]:
        """
        Discovers skills matching an optional query.
        """
        catalog = await self.fetch_marketplace_catalog()
        if not query:
            return catalog

        query = query.lower()
        matched = []
        for skill in catalog:
            name = skill.get("name", "").lower()
            desc = skill.get("description", "").lower()
            if query in name or query in desc:
                matched.append(skill)
        return matched

    async def import_skill(self, skill_def: Dict[str, Any]) -> bool:
        """
        Imports a specific skill definition into the registry.
        """
        name = skill_def.get("name")
        if not name:
            logger.error("Skill definition missing 'name' field.")
            return False

        description = skill_def.get("description", "Imported marketplace skill")
        func = _create_dynamic_skill(skill_def)
        self.registry.register_skill(name=name, func=func, description=description)
        logger.info(f"Successfully imported and registered skill: {name}")
        return True

    async def import_skill_by_name(self, skill_name: str) -> bool:
        """
        Discovers and imports a skill by its exact name.
        """
        catalog = await self.fetch_marketplace_catalog()
        for skill in catalog:
            if skill.get("name") == skill_name:
                return await self.import_skill(skill)
        logger.error(f"Skill '{skill_name}' not found in marketplace catalog.")
        return False
