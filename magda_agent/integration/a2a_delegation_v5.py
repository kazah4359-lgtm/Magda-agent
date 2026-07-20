import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import httpx

from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.integration.a2a_tracing import A2ATracer

logger = logging.getLogger(__name__)


@dataclass
class AgentCardV5:
    """
    Represents the capabilities and identity of an agent in the network, version 5.
    Inspired by trend: A2A (Agent-to-Agent Protocol).
    """
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]
    protocol_version: str = "v5"

    def to_json(self) -> str:
        """
        Serializes the AgentCardV5 to a JSON string.
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCardV5":
        """
        Deserializes an AgentCardV5 from a JSON string.
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


class A2ADiscoveryRegistryV5:
    """
    A registry to manage external agent discovery using Agent Cards (V5).
    Supports capability indexing and parsing cards asynchronously/synchronously.
    """

    def __init__(self) -> None:
        """
        Initializes the A2ADiscoveryRegistryV5.
        """
        self._registry: Dict[str, AgentCardV5] = {}

    def register_agent(self, card: AgentCardV5) -> None:
        """
        Registers an AgentCardV5 in the registry.

        Args:
            card (AgentCardV5): The agent card to register.
        """
        self._registry[card.agent_id] = card
        logger.info(f"Registered AgentCardV5 for agent_id: {card.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregisters an agent from the registry by its ID.

        Args:
            agent_id (str): The ID of the agent to unregister.
        """
        if agent_id in self._registry:
            del self._registry[agent_id]
            logger.info(f"Unregistered agent_id: {agent_id}")
        else:
            logger.warning(f"Attempted to unregister non-existent agent_id: {agent_id}")

    def parse_and_register_cards(self, raw_cards: List[str]) -> List[AgentCardV5]:
        """
        Parses a list of JSON string representations of Agent Cards
        and registers them in the internal store.

        Args:
            raw_cards (List[str]): A list of JSON strings representing AgentCardV5s.

        Returns:
            List[AgentCardV5]: A list of successfully parsed and registered AgentCardV5 objects.
        """
        successfully_parsed: List[AgentCardV5] = []
        for card_json in raw_cards:
            try:
                card = AgentCardV5.from_json(card_json)
                self.register_agent(card)
                successfully_parsed.append(card)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse AgentCardV5. Error: {str(e)}, Raw Data: {card_json}")
        return successfully_parsed

    def get_agent_card(self, agent_id: str) -> Optional[AgentCardV5]:
        """
        Retrieves a registered AgentCardV5 by its agent_id.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            Optional[AgentCardV5]: The AgentCardV5 if found, otherwise None.
        """
        return self._registry.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[AgentCardV5]:
        """
        Returns a list of Agent Cards that support the given capability.

        Args:
            capability (str): The capability to filter by.

        Returns:
            List[AgentCardV5]: A list of matched AgentCardV5 objects.
        """
        matched_agents: List[AgentCardV5] = []
        for agent in self._registry.values():
            if agent.has_capability(capability):
                matched_agents.append(agent)
        return matched_agents

    def get_all_agents(self) -> List[AgentCardV5]:
        """
        Returns a list of all discovered agent cards.

        Returns:
            List[AgentCardV5]: A list of all AgentCardV5 objects in the registry.
        """
        return list(self._registry.values())


class A2ADelegatorV5:
    """
    Handles peer-to-peer task delegation dynamically using Agent Cards (V5) and automatic discovery.
    """

    def __init__(
        self,
        discovery_registry: Optional[A2ADiscoveryRegistryV5] = None,
        security_context: Optional[A2ASecurityContext] = None,
        timeout: float = 10.0
    ) -> None:
        """
        Initializes the A2ADelegatorV5.

        Args:
            discovery_registry: An optional registry to manage agent cards.
            security_context: An optional security context for token generation.
            timeout: The timeout duration for HTTP requests.
        """
        self.discovery_registry = discovery_registry or A2ADiscoveryRegistryV5()
        self.security_context = security_context or A2ASecurityContext()
        self.timeout = timeout

    async def delegate_task(self, peer_agent: AgentCardV5, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates a task to a peer agent asynchronously.

        Args:
            peer_agent (AgentCardV5): The agent card representing the peer.
            task_payload (Dict[str, Any]): The payload containing the task context.

        Returns:
            Dict[str, Any]: The response from the peer agent.
        """
        endpoint = peer_agent.endpoints.get("rpc") or peer_agent.endpoints.get("mcp")
        if not endpoint:
            logger.error(f"Agent {peer_agent.agent_id} missing endpoint")
            raise ValueError(f"Agent {peer_agent.agent_id} missing endpoint")

        headers: Dict[str, str] = {}
        A2ATracer.inject_headers(headers)

        if self.security_context:
            token = self.security_context.generate_token()
            headers["Authorization"] = f"Bearer {token}"
            self.security_context.trace_action("delegate_task_v5", {"target_agent": peer_agent.name})

        A2ATracer.record_event("peer_delegation_v5", {"target_agent_id": peer_agent.agent_id})

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=task_payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to delegate task to {peer_agent.agent_id} at {endpoint}: {e}")
            raise

    async def delegate_by_capability(self, capability: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-discovers an agent supporting the specified capability and delegates the task to it.

        Args:
            capability (str): The capability needed for the task.
            task_payload (Dict[str, Any]): The task details to delegate.

        Returns:
            Dict[str, Any]: The result of the delegation.
        """
        matching_agents = self.discovery_registry.find_agents_by_capability(capability)
        if not matching_agents:
            logger.warning(f"No peer agents found with capability: {capability}")
            raise ValueError(f"No peer agents found supporting capability: {capability}")

        target_agent = matching_agents[0]
        logger.info(f"Auto-discovered target peer agent {target_agent.agent_id} for capability {capability}")
        return await self.delegate_task(target_agent, task_payload)
