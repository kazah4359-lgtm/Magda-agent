import logging
import asyncio
import httpx
import json
import os
from typing import Optional, Any, Dict, List, Union
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.agentskills_importer import AgentSkillsImporter

logger = logging.getLogger(__name__)

class MarketplaceSyncRoutineV4:
    """
    V4 Sync Routine for periodically fetching, parsing, and caching skills
    from an external agentskills.io standard marketplace.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        marketplace_url: str = "https://agentskills.io/api/skills",
        interval: float = 3600.0,
        cache_path: str = ".skill_cache_v4.json"
    ) -> None:
        """
        Initializes the MarketplaceSyncRoutineV4.

        Args:
            registry: The SkillRegistry to sync skills into.
            marketplace_url: The URL of the external marketplace to fetch from.
            interval: How often to sync in seconds.
            cache_path: File path to save/cache fetched skill definitions locally.
        """
        self.registry = registry
        self.marketplace_url = marketplace_url
        self.interval = interval
        self.cache_path = cache_path
        self.importer = AgentSkillsImporter(registry=self.registry)
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    async def fetch_marketplace_data(self) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Fetches the skills data from the marketplace URL and caches it locally.

        Returns:
            The parsed JSON data (list or dict) or None if fetching fails.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.marketplace_url, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                # Save successfully fetched data to local cache
                try:
                    with open(self.cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Successfully cached marketplace data to {self.cache_path}")
                except Exception as cache_err:
                    logger.warning(f"Failed to save local skill cache file: {cache_err}")

                return data
        except Exception as e:
            logger.error(f"Failed to fetch marketplace data from {self.marketplace_url}: {e}")
            return None

    def load_cached_data(self) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Loads cached skill definitions from the local cache file.

        Returns:
            The parsed JSON data or None if unavailable/invalid.
        """
        if not os.path.exists(self.cache_path):
            logger.warning(f"No local cache file found at {self.cache_path}")
            return None

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded skills from local cache file: {self.cache_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load cached skills from {self.cache_path}: {e}")
            return None

    async def run_sync_cycle(self) -> int:
        """
        Runs a complete sync cycle: fetching data (falling back to cache)
        and importing skills into the registry.

        Returns:
            The number of skills successfully imported/registered.
        """
        logger.info(f"Starting marketplace sync cycle from {self.marketplace_url}")

        data = await self.fetch_marketplace_data()
        if not data:
            logger.warning("No fresh data retrieved from marketplace. Attempting local cache fallback...")
            data = self.load_cached_data()

        if not data:
            logger.error("No skill data available (fetch failed and no cache fallback).")
            return 0

        imported_funcs = self.importer.import_skills(data)
        logger.info(f"Marketplace sync complete. Successfully imported {len(imported_funcs)} skills.")
        return len(imported_funcs)

    async def start(self) -> None:
        """
        Starts the background task for periodic synchronization.
        """
        if self._running:
            logger.warning("Marketplace sync routine is already running.")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Marketplace sync routine started with interval {self.interval}s")

    async def stop(self) -> None:
        """
        Stops the background periodic synchronization task.
        """
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Marketplace sync routine stopped.")

    async def _loop(self) -> None:
        """
        Background execution loop.
        """
        while self._running:
            try:
                await self.run_sync_cycle()
            except Exception as e:
                logger.error(f"Error occurred during periodic sync cycle: {e}")

            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
