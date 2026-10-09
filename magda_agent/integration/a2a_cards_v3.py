"""
A2A Agent Discovery Cards V3 Module.

Provides AgentCardV3 representations and A2ADiscoveryCardsV3 discovery mechanism
for resolving, parsing, fetching, and broadcasting agent capability cards across
A2A peer networks.
"""

from dataclasses import asdict, dataclass
import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AgentCardV3:
    """
    Represents the capabilities, endpoints, and identity of an agent in an A2A network (V3 protocol).
    """

    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]
    protocol_version: str = "v3"
    metadata: Optional[Dict[str, Any]] = None
    status: str = "active"

    def __post_init__(self) -> None:
        """
        Validates required fields and data types upon initialization.
        """
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise ValueError("AgentCardV3 requires a non-empty string 'agent_id'")
        if not isinstance(self.capabilities, list):
            raise ValueError("AgentCardV3 'capabilities' must be a list of strings")
        if not isinstance(self.endpoints, dict):
            raise ValueError("AgentCardV3 'endpoints' must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the AgentCardV3 to a Python dictionary.
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Serializes the AgentCardV3 to a JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCardV3":
        """
        Creates an AgentCardV3 instance from a dictionary with validation.
        """
        if not isinstance(data, dict):
            raise ValueError("Expected dictionary payload for AgentCardV3")

        card_data = dict(data)
        # Ensure default values if missing
        if "protocol_version" not in card_data:
            card_data["protocol_version"] = "v3"
        if "status" not in card_data:
            card_data["status"] = "active"

        required_keys = ["agent_id", "capabilities", "endpoints"]
        for key in required_keys:
            if key not in card_data:
                raise ValueError(f"Missing required field '{key}' in AgentCardV3 data")

        # Fallback for optional name and description
        if "name" not in card_data:
            card_data["name"] = card_data["agent_id"]
        if "description" not in card_data:
            card_data["description"] = f"Agent {card_data['agent_id']}"

        return cls(**card_data)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCardV3":
        """
        Deserializes an AgentCardV3 from a JSON string.
        """
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Invalid JSON string for AgentCardV3: {e}")
        return cls.from_dict(data)

    def has_capability(self, capability: str) -> bool:
        """
        Checks if the agent supports the specified capability.
        Supports exact match and prefix match (e.g., 'code' matches 'code_execution' or 'code.execution').
        """
        for cap in self.capabilities:
            if (
                cap == capability
                or cap.startswith(f"{capability}_")
                or cap.startswith(f"{capability}.")
            ):
                return True
        return False

    def matches_any_capability(self, required_capabilities: List[str]) -> bool:
        """
        Returns True if the agent supports at least one of the required capabilities.
        """
        return any(self.has_capability(cap) for cap in required_capabilities)

    def matches_all_capabilities(self, required_capabilities: List[str]) -> bool:
        """
        Returns True if the agent supports all required capabilities.
        """
        return all(self.has_capability(cap) for cap in required_capabilities)


class A2ADiscoveryCardsV3:
    """
    Handles discovery of peer agents using V3 Agent Cards.
    Provides methods to resolve, parse, fetch, cache, search, and broadcast cards.
    """

    def __init__(
        self,
        local_card: Optional[AgentCardV3] = None,
        security_context: Optional[Any] = None,
        timeout: float = 10.0,
    ) -> None:
        self.local_card = local_card
        self.security_context = security_context
        self.timeout = timeout
        self._discovered_agents: Dict[str, AgentCardV3] = {}
        self._capability_index: Dict[str, List[str]] = {}

        if self.local_card:
            self.register_card(self.local_card)

    def register_card(self, card: AgentCardV3) -> None:
        """
        Registers an AgentCardV3 in the local cache and updates the capability index.
        """
        self._discovered_agents[card.agent_id] = card
        for capability in card.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            if card.agent_id not in self._capability_index[capability]:
                self._capability_index[capability].append(card.agent_id)
        logger.info(f"Registered AgentCardV3: {card.agent_id} ({card.name})")

    def parse_card(self, payload: Union[str, Dict[str, Any]]) -> AgentCardV3:
        """
        Parses a JSON string, dictionary, or broadcast envelope into an AgentCardV3 instance.
        """
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON string: {e}")
        elif isinstance(payload, dict):
            data = payload
        else:
            raise ValueError(f"Unsupported payload type for parse_card: {type(payload)}")

        if not isinstance(data, dict):
            raise ValueError("Payload must be a dictionary or JSON object")

        # Envelope check
        if data.get("type") == "a2a_discovery_broadcast":
            version = str(data.get("version", ""))
            if version not in ("3.0", "v3", "3"):
                raise ValueError(f"Unsupported envelope version '{version}' for V3 discovery")
            inner_payload = data.get("payload")
            if isinstance(inner_payload, str):
                return AgentCardV3.from_json(inner_payload)
            elif isinstance(inner_payload, dict):
                return AgentCardV3.from_dict(inner_payload)
            else:
                raise ValueError("Envelope payload is missing or invalid")

        return AgentCardV3.from_dict(data)

    async def fetch_and_parse_cards(
        self, endpoint_url: str, auth_token: Optional[str] = None
    ) -> List[AgentCardV3]:
        """
        Asynchronously fetches JSON-formatted Agent Cards from an A2A endpoint URL,
        validates, parses, registers, and returns them.
        """
        logger.info(f"Fetching V3 Agent Cards from {endpoint_url}")
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(endpoint_url, headers=headers)
            response.raise_for_status()

            body = response.json()
            parsed_cards: List[AgentCardV3] = []

            if isinstance(body, dict):
                # Envelope or single card wrapper
                if "cards" in body and isinstance(body["cards"], list):
                    items = body["cards"]
                else:
                    items = [body]
            elif isinstance(body, list):
                items = body
            else:
                logger.error(f"Unexpected JSON structure from {endpoint_url}: {type(body)}")
                return parsed_cards

            for item in items:
                try:
                    card = self.parse_card(item)
                    self.register_card(card)
                    parsed_cards.append(card)
                except Exception as e:
                    logger.warning(f"Failed to parse card item from {endpoint_url}: {e}")

            return parsed_cards

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentCardV3]:
        """
        Retrieves a registered AgentCardV3 by its ID.
        """
        return self._discovered_agents.get(agent_id)

    def get_all_agents(self) -> List[AgentCardV3]:
        """
        Returns all registered AgentCardV3 instances.
        """
        return list(self._discovered_agents.values())

    def find_agents_by_capability(self, capability: str) -> List[AgentCardV3]:
        """
        Returns all registered AgentCardV3 instances matching the capability.
        """
        return [
            agent
            for agent in self._discovered_agents.values()
            if agent.has_capability(capability)
        ]

    async def resolve_peer_for_capability(
        self, capability: str, exclude_local: bool = True
    ) -> Optional[AgentCardV3]:
        """
        Resolves the best available peer card supporting the required capability.
        """
        matches = self.find_agents_by_capability(capability)
        for card in matches:
            if card.status != "active":
                continue
            if exclude_local and self.local_card and card.agent_id == self.local_card.agent_id:
                continue
            return card
        return None

    async def broadcast_card(self, endpoint_urls: List[str]) -> Dict[str, bool]:
        """
        Broadcasts the local agent's card in V3 envelope format to specified endpoint URLs.
        """
        if not self.local_card:
            raise ValueError("No local card set to broadcast")

        envelope = {
            "type": "a2a_discovery_broadcast",
            "version": "3.0",
            "payload": self.local_card.to_dict(),
        }

        results: Dict[str, bool] = {}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for url in endpoint_urls:
                try:
                    resp = await client.post(url, json=envelope)
                    resp.raise_for_status()
                    results[url] = True
                    logger.info(f"Successfully broadcast card to {url}")
                except Exception as e:
                    logger.error(f"Failed to broadcast card to {url}: {e}")
                    results[url] = False

        return results


# Alias for backward compatibility / alternative imports
A2AAgentDiscoveryV3 = A2ADiscoveryCardsV3
