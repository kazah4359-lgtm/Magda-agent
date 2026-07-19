import logging
from typing import Dict, Any, Optional

from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique

logger = logging.getLogger(__name__)


class A2ADelegatorV5:
    """
    Handles peer-to-peer task delegation using Agent Cards for asynchronous discovery.
    This module delegates tasks to agents dynamically by resolving their capabilities
    using the A2ADiscoveryServiceV3Unique.
    """
    def __init__(self, discovery_service: A2ADiscoveryServiceV3Unique) -> None:
        """
        Initializes the A2ADelegatorV5 with a discovery service.
        """
        self.discovery_service = discovery_service

    async def delegate_task(self, capability: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discovers an agent with the required capability and delegates the task to it asynchronously.

        Args:
            capability (str): The capability required by the task.
            task_payload (Dict[str, Any]): The payload representing the task.

        Returns:
            Dict[str, Any]: The response from the peer agent.

        Raises:
            ValueError: If no agents matching the capability are found.
        """
        agents = self.discovery_service.find_agents_by_capability(capability)
        if not agents:
            logger.error(f"No agents found with capability: {capability}")
            raise ValueError(f"No agents found with capability: {capability}")

        peer_agent = agents[0]

        logger.info(f"Delegating task with capability {capability} to agent {peer_agent.agent_id}")

        return await self.discovery_service.delegate_task(peer_agent.agent_id, task_payload)
