from typing import Dict, List, Optional
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from magda_agent.integration.a2a_security import A2ASecurityContext


@dataclass
class AgentCard:
    """
    Represents the capabilities and identity of an agent in the network.
    Inspired by Google/Linux Foundation A2A Standard.
    """
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]

    def to_json(self) -> str:
        """
        Serializes the AgentCard to a JSON string.
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCard":
        """
        Deserializes an AgentCard from a JSON string.
        """
        return cls(**json.loads(json_str))


class A2ADiscovery:
    """
    Handles discovery of other agents in the network and broadcasting
    the local agent's capabilities.
    """
    def __init__(self, local_card: AgentCard, security_context: Optional[A2ASecurityContext] = None):
        """
        Initializes the discovery module with the local agent's card.
        """
        self.local_card = local_card
        self.security_context = security_context or A2ASecurityContext()
        self._discovered_agents: Dict[str, AgentCard] = {}
        # Indexed by capability for fast lookups
        self._capability_index: Dict[str, List[str]] = {}

    async def broadcast_card(self) -> str:
        """
        Broadcasts the local agent's card to the network.
        In a real scenario, this might use UDP multicast or a central registry API.
        """
        # Mock network broadcast
        logging.info(f"Broadcasting Agent Card: {self.local_card.name}")
        return self.local_card.to_json()

    async def fetch_cards(self, mock_network_cards: Optional[List[str]] = None, auth_token: Optional[str] = None) -> None:
        """
        Fetches Agent Cards from the network and indexes them.
        Requires an auth_token for secure discovery.
        """
        if auth_token and not self.security_context.validate_token(auth_token):
            logging.error("Invalid auth token for fetch_cards")
            raise ValueError("Invalid authentication token")

        self.security_context.trace_action("fetch_cards", {"count": len(mock_network_cards) if mock_network_cards else 0})

        if mock_network_cards is None:
            # Simulate fetching from network
            mock_network_cards = []

        for card_json in mock_network_cards:
            try:
                card = AgentCard.from_json(card_json)
                self._register_agent(card)
            except Exception as e:
                logging.error(f"Failed to parse Agent Card: {e}")

    def _register_agent(self, card: AgentCard) -> None:
        """
        Registers a discovered agent internally and updates the capability index.
        """
        self._discovered_agents[card.agent_id] = card
        for capability in card.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            if card.agent_id not in self._capability_index[capability]:
                self._capability_index[capability].append(card.agent_id)
        logging.info(f"Discovered Agent: {card.name} with capabilities {card.capabilities}")

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentCard]:
        """
        Retrieves a discovered agent's card by its ID.
        """
        return self._discovered_agents.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[AgentCard]:
        """
        Returns a list of Agent Cards that support the given capability.
        """
        agent_ids = self._capability_index.get(capability, [])
        return [self._discovered_agents[aid] for aid in agent_ids]
