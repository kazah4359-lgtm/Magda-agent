from typing import Dict, Any, List, Optional, Union
import logging
from magda_agent.integration.a2a_discovery import AgentCard, A2ADiscovery
from magda_agent.integration.a2a_discovery_v2 import AgentCardV2, A2ADiscoveryV2
from magda_agent.integration.a2a_delegation import A2ADelegator

class A2AManager:
    """
    Orchestrates the discovery of other agents via Agent Cards and the delegation
    of sub-plans/tasks to capable peers in a peer-to-peer network.
    Inspired by A2A Protocol trends.
    """
    def __init__(self, local_card: Union[AgentCard, AgentCardV2]) -> None:
        """
        Initializes the manager with the local agent's identity and capabilities.

        Args:
            local_card: The AgentCard or AgentCardV2 representing this agent.
        """
        if isinstance(local_card, AgentCardV2):
            self.discovery = A2ADiscoveryV2(local_card=local_card) # type: ignore
        else:
            self.discovery = A2ADiscovery(local_card=local_card) # type: ignore

        self.delegator = A2ADelegator(discovery=self.discovery) # type: ignore

    async def start(self) -> str:
        """
        Starts the manager by broadcasting the local agent's capabilities to the network.

        Returns:
            The JSON representation of the broadcasted AgentCard.
        """
        logging.info("Starting A2AManager and broadcasting local capabilities...")
        return await self.discovery.broadcast_card()

    async def discover_peers(self, mock_network_cards: Optional[List[str]] = None) -> None:
        """
        Discovers peers by fetching their Agent Cards from the network.

        Args:
            mock_network_cards: Optional list of JSON strings representing mocked Agent Cards.
        """
        logging.info("A2AManager discovering peers...")
        if isinstance(self.discovery, A2ADiscoveryV2):
            await self.discovery.fetch_cards(network_envelopes=mock_network_cards)
        else:
            await self.discovery.fetch_cards(mock_network_cards=mock_network_cards)

    def get_known_peers(self) -> Union[List[AgentCard], List[AgentCardV2]]:
        """
        Retrieves all currently known peers discovered in the network.

        Returns:
            A list of discovered AgentCard objects.
        """
        return list(self.discovery._discovered_agents.values())

    async def delegate_task(self, capability: str, task_context: Dict[str, Any]) -> str:
        """
        Delegates a task to a discovered peer that supports the required capability.

        Args:
            capability: The required capability (e.g., 'code_execution').
            task_context: The context or sub-plan of the task to delegate.

        Returns:
            A string indicating the outcome of the delegation.
        """
        logging.info(f"A2AManager attempting to delegate task requiring capability: {capability}")
        return await self.delegator.delegate_subplan(capability, task_context)
