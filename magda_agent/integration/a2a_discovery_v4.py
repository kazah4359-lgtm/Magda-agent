import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import httpx

@dataclass
class AgentCardV4:
    """
    Represents the capabilities and identity of an agent in the network, version 4.
    Inspired by trend: A2A (Agent-to-Agent Protocol).
    """
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]

    def to_json(self) -> str:
        """
        Serializes the AgentCardV4 to a JSON string.
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCardV4":
        """
        Deserializes an AgentCardV4 from a JSON string.
        """
        data = json.loads(json_str)
        return cls(**data)


class A2ADiscoveryRegistryV4:
    """
    A registry to manage external agent discovery using Agent Cards (V4).
    """

    def __init__(self) -> None:
        """
        Initializes the A2ADiscoveryRegistryV4.
        """
        self._registry: Dict[str, AgentCardV4] = {}

    def register_agent(self, card: AgentCardV4) -> None:
        """
        Registers an AgentCardV4 in the registry.

        Args:
            card (AgentCardV4): The agent card to register.
        """
        self._registry[card.agent_id] = card
        logging.info(f"Registered AgentCardV4 for agent_id: {card.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregisters an agent from the registry by its ID.

        Args:
            agent_id (str): The ID of the agent to unregister.
        """
        if agent_id in self._registry:
            del self._registry[agent_id]
            logging.info(f"Unregistered agent_id: {agent_id}")
        else:
            logging.warning(f"Attempted to unregister non-existent agent_id: {agent_id}")

    def parse_and_register_cards(self, raw_cards: List[str]) -> List[AgentCardV4]:
        """
        Parses a list of JSON string representations of Agent Cards
        and registers them in the internal store.

        Args:
            raw_cards (List[str]): A list of JSON strings representing AgentCardV4s.

        Returns:
            List[AgentCardV4]: A list of successfully parsed and registered AgentCardV4 objects.
        """
        successfully_parsed: List[AgentCardV4] = []
        for card_json in raw_cards:
            try:
                card = AgentCardV4.from_json(card_json)
                self.register_agent(card)
                successfully_parsed.append(card)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                logging.error(f"Failed to parse AgentCardV4. Error: {str(e)}, Raw Data: {card_json}")
        return successfully_parsed

    def get_agent_card(self, agent_id: str) -> Optional[AgentCardV4]:
        """
        Retrieves a registered AgentCardV4 by its agent_id.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            Optional[AgentCardV4]: The AgentCardV4 if found, otherwise None.
        """
        return self._registry.get(agent_id)

    def get_all_agents(self) -> List[AgentCardV4]:
        """
        Returns a list of all discovered agent cards.

        Returns:
            List[AgentCardV4]: A list of all AgentCardV4 objects in the registry.
        """
        return list(self._registry.values())


class AgentCardBroadcasterV4:
    """
    Component that periodically or on-demand broadcasts the agent's capabilities
    via Agent Card formatting over the network.
    Inspired by A2A standard trend V4.
    """

    def __init__(
        self,
        agent_card: AgentCardV4,
        endpoints: Optional[List[str]] = None,
        broadcast_interval: float = 30.0,
    ) -> None:
        """
        Initializes the broadcaster.

        Args:
            agent_card (AgentCardV4): The AgentCardV4 representing this agent.
            endpoints (Optional[List[str]]): List of HTTP endpoints to broadcast to.
            broadcast_interval (float): Interval in seconds between periodic broadcasts.
        """
        self.agent_card = agent_card
        self.endpoints = endpoints or []
        self.broadcast_interval = broadcast_interval
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running: bool = False

    def generate_broadcast_payload(self) -> Dict[str, Any]:
        """
        Generates the standard envelope payload containing the Agent Card V4 and capabilities.

        Returns:
            Dict[str, Any]: The envelope dictionary.
        """
        card_data = asdict(self.agent_card)
        payload = {
            "type": "a2a_discovery_broadcast",
            "version": "4.0",
            "agent_card": card_data,
            "capabilities": self.agent_card.capabilities,
            "agent_id": self.agent_card.agent_id,
        }
        return payload

    async def broadcast_once(self, target_endpoints: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Asynchronously broadcasts the agent card payload to network endpoints.

        Args:
            target_endpoints (Optional[List[str]]): Specific endpoints to broadcast to.
                If None, defaults to self.endpoints.

        Returns:
            Dict[str, bool]: A mapping of endpoint URL to broadcast success status.
        """
        endpoints = target_endpoints if target_endpoints is not None else self.endpoints
        if not endpoints:
            logging.warning("No endpoints configured for AgentCardBroadcasterV4.")
            return {}

        payload = self.generate_broadcast_payload()
        results: Dict[str, bool] = {}

        async with httpx.AsyncClient() as client:
            for url in endpoints:
                try:
                    response = await client.post(url, json=payload, timeout=5.0)
                    response.raise_for_status()
                    logging.info(f"Successfully broadcasted AgentCardV4 to {url}")
                    results[url] = True
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    logging.error(f"Failed to broadcast AgentCardV4 to {url}: {e}")
                    results[url] = False
                except Exception as e:
                    logging.error(f"Unexpected error broadcasting AgentCardV4 to {url}: {e}")
                    results[url] = False

        return results

    async def start_periodic_broadcast(self, interval: Optional[float] = None) -> None:
        """
        Starts periodic background broadcasting of the agent card.

        Args:
            interval (Optional[float]): Override broadcast interval in seconds.
        """
        if self._running:
            logging.warning("Periodic broadcast is already running.")
            return

        if interval is not None:
            self.broadcast_interval = interval

        self._running = True
        self._broadcast_task = asyncio.create_task(self._periodic_loop())
        logging.info(f"Started periodic AgentCardV4 broadcast every {self.broadcast_interval}s")

    async def _periodic_loop(self) -> None:
        """Background loop for periodic broadcasting."""
        while self._running:
            try:
                await self.broadcast_once()
            except Exception as e:
                logging.error(f"Error in periodic broadcast loop: {e}")

            try:
                await asyncio.sleep(self.broadcast_interval)
            except asyncio.CancelledError:
                break

    async def stop_periodic_broadcast(self) -> None:
        """
        Stops periodic background broadcasting.
        """
        if not self._running:
            return

        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None
        logging.info("Stopped periodic AgentCardV4 broadcast")


# Alias for backward/naming compatibility
A2AAgentCardBroadcasterV4 = AgentCardBroadcasterV4
