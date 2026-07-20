import asyncio
import logging
import httpx
from typing import Optional, Any, Dict, List
from magda_agent.skills.registry import SkillRegistry
from magda_agent.skills.agentskills_importer import AgentSkillsImporter

logger = logging.getLogger(__name__)

class MarketplaceSyncRoutineV4:
    """
    Routine for fetching, parsing, and synchronizing skills from an external marketplace periodically.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        marketplace_url: str = "https://agentskills.io/api/skills",
        sync_interval_seconds: int = 3600
    ) -> None:
        """
        Initializes the MarketplaceSyncRoutineV4.

        Args:
            registry: The SkillRegistry to sync skills into.
            marketplace_url: The URL of the external marketplace to fetch from.
            sync_interval_seconds: The interval in seconds to fetch the data.
        """
        self.registry = registry
        self.marketplace_url = marketplace_url
        self.sync_interval_seconds = sync_interval_seconds
        self.importer = AgentSkillsImporter(registry=self.registry)
        self._sync_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

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

    async def _periodic_sync_loop(self) -> None:
        """
        The background loop running the periodic syncs.
        """
        while not self._stop_event.is_set():
            await self.run_sync_cycle()
            try:
                # Wait for the interval or until stop is requested
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.sync_interval_seconds)
            except asyncio.TimeoutError:
                pass  # Timeout means we should run again
            except asyncio.CancelledError:
                break # Task was cancelled

    def start_periodic_sync(self) -> None:
        """
        Starts the periodic synchronization in the background.
        """
        if self._sync_task is not None and not self._sync_task.done():
            logger.warning("Periodic sync is already running.")
            return

        self._stop_event.clear()
        self._sync_task = asyncio.create_task(self._periodic_sync_loop())
        logger.info("Marketplace periodic sync started.")

    async def stop_periodic_sync(self) -> None:
        """
        Stops the periodic synchronization.
        """
        if self._sync_task is None or self._sync_task.done():
            logger.warning("Periodic sync is not running.")
            return

        self._stop_event.set()
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
            logger.info("Marketplace periodic sync stopped.")
