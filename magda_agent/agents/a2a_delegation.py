from typing import Dict, Any, List, Optional
import logging
from magda_agent.integration.a2a_discovery import AgentCard
from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique
from magda_agent.integration.a2a_cards import AgentCardV3

class A2ADelegationAgent:
    """
    A high-level agent responsible for A2A (Agent-to-Agent) operations.
    It manages discovery of peers and delegation of subtasks.
    Inspired by the A2A Protocol trend (June 2026).
    """

    def __init__(self, local_card: Optional[AgentCardV3] = None) -> None:
        """
        Initializes the A2ADelegationAgent.

        Args:
            local_card: The AgentCard representing this agent's identity and capabilities.
        """
        self.discovery_service = A2ADiscoveryServiceV3Unique()
        self.local_card = local_card
        if local_card:
            logging.info(f"A2ADelegationAgent initialized for agent: {local_card.name}")

    async def discover_peers(self, raw_cards: List[str]) -> List[AgentCardV3]:
        """
        Discovers other agents in the network by parsing their Agent Cards.

        Args:
            raw_cards: A list of JSON strings representing AgentCards.

        Returns:
            A list of successfully parsed AgentCardV3 objects.
        """
        peers = self.discovery_service.parse_and_register_cards(raw_cards)
        logging.info(f"A2ADelegationAgent discovered {len(peers)} peers.")
        return peers

    async def delegate_task_by_capability(self, capability: str, task: Dict[str, Any]) -> str:
        """
        Delegates a task to a peer that has the required capability.

        Args:
            capability: The required capability for the task (e.g., 'data_analysis').
            task: The task details or context to be delegated.

        Returns:
            A result string describing the delegation outcome and the target agent.
        """
        logging.info(f"A2ADelegationAgent attempting to delegate task requiring '{capability}'")
        agents = self.discovery_service.find_agents_by_capability(capability)

        if not agents:
            logging.warning(f"No agents found for capability: {capability}")
            return "No agent found"

        # Select the first available agent
        target_agent = agents[0]

        try:
            result = await self.discovery_service.delegate_task(target_agent.agent_id, task)
            return f"Delegated to Agent {target_agent.name}: {result.get('status', 'Success')}"
        except Exception as e:
            logging.error(f"Delegation to {target_agent.name} failed: {e}")
            return f"Delegation to {target_agent.name} failed: {e}"
