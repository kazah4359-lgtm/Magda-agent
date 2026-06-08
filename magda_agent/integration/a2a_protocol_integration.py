from typing import Dict, Any, List, Optional
import logging
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_protocol import A2AProtocolManager

class A2AProtocolIntegration:
    """
    Integration layer for the A2A Protocol, wrapping A2AProtocolManager to provide
    an interface for discovering peers via Agent Cards and delegating peer-to-peer tasks.
    """
    def __init__(self, manager: A2AProtocolManager) -> None:
        """
        Initializes the A2A Protocol Integration.

        Args:
            manager: An instance of A2AProtocolManager.
        """
        self.manager = manager

    async def start_and_discover(self, mock_network_cards: Optional[List[str]] = None) -> List[AgentCard]:
        """
        Starts the local manager to broadcast its card and discovers peers.

        Args:
            mock_network_cards: Optional list of raw JSON card payloads for mock discovery.

        Returns:
            A list of discovered AgentCard objects.
        """
        logging.info("A2AProtocolIntegration: Starting and discovering peers...")
        await self.manager.start()
        await self.manager.discover_peers(mock_network_cards=mock_network_cards)
        return self.manager.get_known_peers()

    async def execute_task_with_peer(self, capability: str, task_context: Dict[str, Any]) -> str:
        """
        Delegates a peer-to-peer task using the protocol manager.

        Args:
            capability: The capability required to perform the task.
            task_context: Context for the task to be delegated.

        Returns:
            A result string describing the delegation success or failure.
        """
        logging.info(f"A2AProtocolIntegration: Executing task requiring '{capability}' with peer...")
        return await self.manager.delegate_task(capability, task_context)
