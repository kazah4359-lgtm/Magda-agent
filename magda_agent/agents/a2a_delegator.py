import logging
from typing import Dict, Any, List, Optional
from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_discovery_v3_unique import A2ADiscoveryServiceV3Unique

class A2ADelegatorSubAgent:
    """
    A subagent capable of peer-to-peer task delegation using the A2A standard.
    It registers its own identity via an Agent Card, parses and registers peer cards,
    discovers peers by their capabilities, and delegates tasks asynchronously.
    """

    def __init__(
        self,
        agent_id: str = "a2a-delegator-subagent",
        name: str = "A2A Delegator Subagent",
        capabilities: Optional[List[str]] = None,
        discovery_service: Optional[A2ADiscoveryServiceV3Unique] = None
    ) -> None:
        """
        Initializes the A2ADelegatorSubAgent.

        Args:
            agent_id: The unique identifier for this subagent.
            name: The name of this subagent.
            capabilities: A list of capabilities supported by this subagent.
            discovery_service: An optional external A2ADiscoveryServiceV3Unique instance.
        """
        self.agent_id: str = agent_id
        self.name: str = name
        self.capabilities: List[str] = capabilities or ["delegation", "peer-coordination"]
        self.discovery_service: A2ADiscoveryServiceV3Unique = discovery_service or A2ADiscoveryServiceV3Unique()
        self.local_card: AgentCardV3 = AgentCardV3(
            agent_id=self.agent_id,
            name=self.name,
            description=f"A2A Delegator Subagent: {self.name}",
            capabilities=self.capabilities,
            endpoints={"rpc": f"http://{self.agent_id}/rpc"}
        )
        logging.info(f"A2ADelegatorSubAgent '{self.name}' initialized successfully.")

    def get_local_card(self) -> AgentCardV3:
        """
        Retrieves the AgentCard representing this subagent's identity and capabilities.

        Returns:
            The local AgentCardV3 instance.
        """
        return self.local_card

    def discover_peers(self, raw_cards: List[str]) -> List[AgentCardV3]:
        """
        Discovers and registers peer agents from their raw JSON Agent Card strings.

        Args:
            raw_cards: A list of JSON strings representing peer Agent Cards.

        Returns:
            A list of successfully parsed and registered AgentCardV3 objects.
        """
        logging.info(f"A2ADelegatorSubAgent '{self.name}' parsing {len(raw_cards)} peer cards.")
        parsed_cards = self.discovery_service.parse_and_register_cards(raw_cards)
        return parsed_cards

    def find_peers_by_capability(self, capability: str) -> List[AgentCardV3]:
        """
        Finds registered peer agents that support a specific capability.

        Args:
            capability: The capability to search for.

        Returns:
            A list of matching AgentCardV3 objects.
        """
        logging.info(f"A2ADelegatorSubAgent '{self.name}' searching for peers with capability: {capability}")
        return self.discovery_service.find_agents_by_capability(capability)

    async def delegate_task(self, peer_agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates a task directly to a specific peer agent by ID.

        Args:
            peer_agent_id: The target peer's unique identifier.
            task: The task payload to delegate.

        Returns:
            A dictionary containing the response or status of the delegation.
        """
        logging.info(f"A2ADelegatorSubAgent '{self.name}' delegating task to agent ID: {peer_agent_id}")
        try:
            result = await self.discovery_service.delegate_task(peer_agent_id, task)
            return result
        except Exception as e:
            logging.error(f"Delegation to agent {peer_agent_id} failed: {e}")
            return {"status": "error", "message": str(e)}

    async def delegate_task_by_capability(self, capability: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finds a peer supporting the required capability and delegates a task to them.

        Args:
            capability: The required capability.
            task: The task payload to delegate.

        Returns:
            A dictionary containing the delegation result or an error message if no peer is found.
        """
        logging.info(f"A2ADelegatorSubAgent '{self.name}' attempting delegation by capability: {capability}")
        peers = self.find_peers_by_capability(capability)
        if not peers:
            logging.warning(f"No peers found with capability: {capability}")
            return {"status": "error", "message": f"No peer agent found supporting capability: {capability}"}

        # Select the first matching peer agent
        target_peer = peers[0]
        return await self.delegate_task(target_peer.agent_id, task)
