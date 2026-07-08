from typing import Dict, List, Optional, Any
import json
import logging
import httpx
from magda_agent.integration.a2a_cards import AgentCardV3


class A2ADiscoveryServiceV3Unique:
    """
    Implements A2A Agent Discovery Service logic for v3.
    This class handles parsing of AgentCards, registering them, finding peers by capabilities,
    and delegating tasks to peers asynchronously.
    """
    def __init__(self) -> None:
        """
        Initializes the A2ADiscoveryServiceV3Unique registry.
        """
        self._discovered_agents: Dict[str, AgentCardV3] = {}

    def parse_and_register_cards(self, raw_cards: List[str]) -> List[AgentCardV3]:
        """
        Parses a list of JSON string representations of Agent Cards
        and registers them in the internal store.

        Args:
            raw_cards: A list of JSON strings representing AgentCards.

        Returns:
            A list of successfully parsed AgentCardV3 objects.
        """
        successfully_parsed: List[AgentCardV3] = []
        for card_json in raw_cards:
            try:
                card = AgentCardV3.from_json(card_json)
                self._discovered_agents[card.agent_id] = card
                successfully_parsed.append(card)
                logging.info(f"Successfully parsed and registered AgentCard for agent_id: {card.agent_id}")
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                logging.error(f"Failed to parse Agent Card. Error: {str(e)}, Raw Data: {card_json}")

        return successfully_parsed

    def get_agent_card(self, agent_id: str) -> Optional[AgentCardV3]:
        """
        Retrieves a registered AgentCardV3 by its agent_id.

        Args:
            agent_id: The ID of the agent to retrieve.

        Returns:
            The AgentCardV3 if found, else None.
        """
        return self._discovered_agents.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[AgentCardV3]:
        """
        Returns a list of Agent Cards that support the given capability.
        Supports capability matching logic from AgentCardV3.

        Args:
            capability: The capability string to match.

        Returns:
            A list of matched AgentCardV3 objects.
        """
        matched_agents: List[AgentCardV3] = []
        for agent in self._discovered_agents.values():
            if agent.has_capability(capability):
                matched_agents.append(agent)
        return matched_agents

    async def delegate_task(self, peer_agent_id: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates a task to a peer agent asynchronously using httpx.

        Args:
            peer_agent_id: The ID of the peer agent to delegate to.
            task_payload: The dictionary payload describing the task.

        Returns:
            A dictionary containing the response or status of the delegation.

        Raises:
            ValueError: If the peer agent is not found or has no RPC endpoint.
            httpx.HTTPError: If the HTTP request fails.
        """
        agent = self.get_agent_card(peer_agent_id)
        if not agent:
            logging.error(f"Cannot delegate to unknown agent: {peer_agent_id}")
            raise ValueError(f"Agent not found: {peer_agent_id}")

        rpc_endpoint = agent.endpoints.get("rpc")
        if not rpc_endpoint:
            logging.error(f"Agent {peer_agent_id} has no RPC endpoint")
            raise ValueError(f"Agent {peer_agent_id} has no RPC endpoint")

        logging.info(f"Delegating task to {peer_agent_id} at {rpc_endpoint}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{rpc_endpoint}/delegate",
                json=task_payload
            )
            response.raise_for_status()
            return response.json()
