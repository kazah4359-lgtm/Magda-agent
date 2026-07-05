from typing import Dict, List, Optional
import json
import logging
from magda_agent.integration.a2a_discovery import AgentCard

class A2ADiscoveryV3:
    """
    Implements A2A Agent Discovery v3 logic using the new standard.
    This class handles the parsing and management of AgentCards.
    """
    def __init__(self):
        """
        Initializes the A2ADiscoveryV3 registry.
        """
        self._discovered_agents: Dict[str, AgentCard] = {}

    def parse_and_register_cards(self, raw_cards: List[str]) -> List[AgentCard]:
        """
        Parses a list of JSON string representations of Agent Cards
        and registers them in the internal store.

        Args:
            raw_cards: A list of JSON strings representing AgentCards.

        Returns:
            A list of successfully parsed AgentCard objects.
        """
        successfully_parsed = []
        for card_json in raw_cards:
            try:
                card = AgentCard.from_json(card_json)
                self._discovered_agents[card.agent_id] = card
                successfully_parsed.append(card)
                logging.info(f"Successfully parsed and registered AgentCard for agent_id: {card.agent_id}")
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                logging.error(f"Failed to parse Agent Card. Error: {str(e)}, Raw Data: {card_json}")

        return successfully_parsed

    def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        """
        Retrieves a registered AgentCard by its agent_id.
        """
        return self._discovered_agents.get(agent_id)

    def get_all_agents(self) -> List[AgentCard]:
        """
        Returns a list of all discovered agents.
        """
        return list(self._discovered_agents.values())
