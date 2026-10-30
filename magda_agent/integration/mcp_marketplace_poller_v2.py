import asyncio
import logging
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


class MCPMarketplacePollerV2:
    """
    Background worker job that periodically syncs with an agentskills.io marketplace
    or endpoint to dynamically discover and register MCP-compatible tools.
    """

    def __init__(
        self,
        registry: Any,
        scheduler: Any,
        marketplace_url: str = "https://api.agentskills.io/v1/tools",
        cron_expr: str = "0 * * * *",
    ) -> None:
        """
        Initialize the MCPMarketplacePollerV2.

        Args:
            registry: Registry instance into which valid tools/skills will be loaded.
            scheduler: CronScheduler instance to schedule the polling task.
            marketplace_url: Marketplace URL endpoint to fetch agentskills.io tool definitions.
            cron_expr: Cron pattern for polling frequency (default hourly: "0 * * * *").
        """
        self.registry = registry
        self.scheduler = scheduler
        self.marketplace_url = marketplace_url
        self.cron_expr = cron_expr
        self.client = httpx.AsyncClient()

    def schedule_sync(self) -> None:
        """
        Registers the sync_marketplace task with the scheduled cron task runner.
        """
        if hasattr(self.scheduler, "add_task"):
            self.scheduler.add_task("mcp_marketplace_sync_v2", self.cron_expr, self.sync_marketplace)
        elif hasattr(self.scheduler, "register_task"):
            self.scheduler.register_task("mcp_marketplace_sync_v2", self.cron_expr, self.sync_marketplace)
        logger.info(f"Scheduled MCP Marketplace V2 sync for {self.marketplace_url} with cron {self.cron_expr}")

    def _extract_tools_list(self, data: Any) -> Optional[List[Dict[str, Any]]]:
        """
        Extracts a list of tool/skill dictionaries from various agentskills.io JSON formats.
        Supports direct list format as well as wrapped dicts with 'skills', 'tools', or 'data' keys.
        """
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("skills", "tools", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return [item for item in data[key] if isinstance(item, dict)]
        return None

    async def sync_marketplace(self) -> int:
        """
        Fetches agentskills.io tool definitions from the marketplace and registers them into the registry.

        Returns:
            int: The count of tools successfully registered.
        """
        logger.info(f"Starting MCP Marketplace V2 sync from {self.marketplace_url}...")
        try:
            response = await self.client.get(self.marketplace_url, timeout=10.0)
            response.raise_for_status()

            data = response.json()
            tools_list = self._extract_tools_list(data)

            if tools_list is None:
                logger.warning(
                    f"Marketplace sync failed: Expected JSON list or wrapped object, got invalid format: {type(data)}"
                )
                return 0

            registered_count = 0
            for item in tools_list:
                # Validate minimum required fields for agentskills.io standard
                has_name = "name" in item
                has_desc = "description" in item
                has_params = "parameters" in item or "inputSchema" in item or "schema" in item

                if not (has_name and has_desc and has_params):
                    logger.debug(f"Skipping malformed tool item missing required fields: {item}")
                    continue

                # Normalization for registry compatibility
                tool_payload = dict(item)
                if "parameters" in tool_payload and "inputSchema" not in tool_payload:
                    tool_payload["inputSchema"] = tool_payload["parameters"]
                elif "inputSchema" in tool_payload and "parameters" not in tool_payload:
                    tool_payload["parameters"] = tool_payload["inputSchema"]

                # Register in registry using available method
                registered = False
                if hasattr(self.registry, "load_tool"):
                    registered = bool(self.registry.load_tool(tool_payload))
                elif hasattr(self.registry, "register_tool"):
                    registered = bool(self.registry.register_tool(tool_payload))
                elif hasattr(self.registry, "register_skill"):
                    registered = bool(self.registry.register_skill(tool_payload))
                elif hasattr(self.registry, "load_skill"):
                    registered = bool(self.registry.load_skill(tool_payload))

                if registered:
                    registered_count += 1

            logger.info(f"MCP Marketplace V2 sync complete. Registered {registered_count} tools.")
            return registered_count

        except httpx.HTTPError as e:
            logger.error(f"Network error during MCP Marketplace V2 sync: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during MCP Marketplace V2 sync: {e}", exc_info=True)
            return 0

    async def close(self) -> None:
        """Close the internal HTTP client."""
        await self.client.aclose()
