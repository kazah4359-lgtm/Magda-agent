import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

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
