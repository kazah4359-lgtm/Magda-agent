from typing import Any, Dict, Optional
import aiohttp
import logging

logger = logging.getLogger(__name__)

class A2ADelegator:
    """
    Handles delegating tasks to other agents via the A2A protocol.
    """
    def __init__(self, peer_registry_url: Optional[str] = None):
        self.peer_registry_url = peer_registry_url

    async def delegate_task(self, task_description: str, peer_url: str) -> Dict[str, Any]:
        """
        Delegates a task to a specific peer agent.

        Args:
            task_description: The description of the task to delegate.
            peer_url: The API endpoint of the peer agent.

        Returns:
            The response from the peer agent.
        """
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "execute_task",
                    "params": {"task": task_description},
                    "id": 1
                }
                async with session.post(peer_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.info(f"Successfully delegated task to {peer_url}")
                    return result
        except Exception as e:
            logger.error(f"Failed to delegate task to {peer_url}: {e}")
            return {"error": str(e), "status": "failed"}
