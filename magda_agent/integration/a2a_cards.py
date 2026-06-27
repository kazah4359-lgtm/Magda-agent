from typing import Dict, List, Optional
import json
import logging
from dataclasses import dataclass, asdict
from magda_agent.integration.a2a_security import A2ASecurityContext


@dataclass
class AgentCardV3:
    """
    Represents the capabilities and identity of an agent in the network, version 3.
    Supports standardized capability matching.
    """
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]
    protocol_version: str = "v3"

    def to_json(self) -> str:
        """
        Serializes the AgentCardV3 to a JSON string.
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCardV3":
        """
        Deserializes an AgentCardV3 from a JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)

    def has_capability(self, capability: str) -> bool:
        """
        Checks if the agent has the specified capability.
        Supports exact match and prefix match (e.g. 'code' matches 'code_execution').
        """
        for cap in self.capabilities:
            if cap == capability or cap.startswith(f"{capability}_"):
                return True
        return False

    def matches_any_capability(self, required_capabilities: List[str]) -> bool:
        """
        Checks if the agent matches any of the required capabilities.
        """
        return any(self.has_capability(cap) for cap in required_capabilities)


class A2ADiscoveryV3:
    """
    Handles discovery of other agents in the network and broadcasting
    the local agent's capabilities using the v3 protocol.
    """
    def __init__(self, local_card: AgentCardV3, security_context: Optional[A2ASecurityContext] = None) -> None:
        """
        Initializes the discovery module with the local agent's card.
        """
        self.local_card = local_card
        self.security_context = security_context or A2ASecurityContext()
        self._discovered_agents: Dict[str, AgentCardV3] = {}
        self._capability_index: Dict[str, List[str]] = {}

    async def broadcast_card(self) -> str:
        """
        Broadcasts the local agent's card to the network in a v3 envelope format.
        """
        logging.info(f"Broadcasting Agent Card V3: {self.local_card.name}")

        envelope = {
            "type": "a2a_discovery_broadcast",
            "version": "3.0",
            "payload": asdict(self.local_card)
        }
        return json.dumps(envelope)

    async def fetch_cards(self, network_envelopes: Optional[List[str]] = None, auth_token: Optional[str] = None) -> None:
        """
        Fetches Agent Cards from the network envelopes and indexes them.
        """
        if auth_token and not self.security_context.validate_token(auth_token):
            logging.error("Invalid auth token for fetch_cards")
            raise ValueError("Invalid authentication token")

        self.security_context.trace_action("fetch_cards_v3", {"count": len(network_envelopes) if network_envelopes else 0})

        if network_envelopes is None:
            network_envelopes = []

        for envelope_json in network_envelopes:
            try:
                envelope = json.loads(envelope_json)
                if envelope.get("type") == "a2a_discovery_broadcast" and envelope.get("version") == "3.0":
                    payload = envelope.get("payload")
                    if isinstance(payload, str):
                        card = AgentCardV3.from_json(payload)
                    else:
                        card = AgentCardV3(**payload)
                    self._register_agent(card)
            except Exception as e:
                logging.error(f"Failed to parse Agent Card Envelope V3: {e}")

    def _register_agent(self, card: AgentCardV3) -> None:
        """
        Registers a discovered agent internally and updates the capability index.
        """
        self._discovered_agents[card.agent_id] = card
        for capability in card.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            if card.agent_id not in self._capability_index[capability]:
                self._capability_index[capability].append(card.agent_id)
        logging.info(f"Discovered Agent V3: {card.name} with capabilities {card.capabilities}")

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentCardV3]:
        """
        Retrieves a discovered agent's card by its ID.
        """
        return self._discovered_agents.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[AgentCardV3]:
        """
        Returns a list of Agent Cards that support the given capability.
        Supports capability matching logic.
        """
        matched_agents = []
        for agent in self._discovered_agents.values():
            if agent.has_capability(capability):
                matched_agents.append(agent)
        return matched_agents
