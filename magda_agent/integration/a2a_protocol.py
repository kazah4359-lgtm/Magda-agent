from typing import Dict, Any, List, Optional
import logging
from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery
from magda_agent.integration.a2a_delegation import A2ADelegator

class A2AProtocolManager:
    """
    Orchestrates the discovery of other agents via Agent Cards and the delegation
    of sub-plans/tasks to capable peers in a peer-to-peer network.
    """
    def __init__(self, local_card: AgentCard) -> None:
        """
        Initializes the A2A Protocol Manager with local capabilities.

        Args:
            local_card: The agent card for the current agent.
        """
        self.discovery = A2ADiscovery(local_card=local_card)
        self.delegator = A2ADelegator(discovery=self.discovery)

    async def start(self) -> str:
        """
        Starts the protocol manager by broadcasting the local card.

        Returns:
            A string containing the broadcasted JSON payload.
        """
        logging.info("Starting A2AProtocolManager and broadcasting local capabilities...")
        return await self.discovery.broadcast_card()

    async def discover_peers(self, mock_network_cards: Optional[List[str]] = None) -> None:
        """
        Discovers peers on the network.

        Args:
            mock_network_cards: Optional list of raw JSON card payloads.
        """
        logging.info("A2AProtocolManager discovering peers...")
        await self.discovery.fetch_cards(mock_network_cards=mock_network_cards)

    def get_known_peers(self) -> List[AgentCard]:
        """
        Gets the list of currently known peers.

        Returns:
            A list of AgentCard objects.
        """
        return list(self.discovery._discovered_agents.values())

    async def delegate_task(self, capability: str, task_context: Dict[str, Any]) -> str:
        """
        Delegates a task context to a peer supporting the required capability.

        Args:
            capability: The capability required to perform the task.
            task_context: Context for the task to be delegated.

        Returns:
            A result string describing the delegation success or failure.
        """
        logging.info(f"A2AProtocolManager attempting to delegate task requiring capability: {capability}")
        return await self.delegator.delegate_subplan(capability, task_context)
