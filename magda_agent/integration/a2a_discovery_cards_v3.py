from typing import List
import logging
import httpx
from magda_agent.integration.a2a_cards import AgentCardV3

class A2ADiscoveryCardsV3:
    """
    Handles discovery of other agents in the network by fetching and parsing Agent Cards
    from A2A discovery endpoints using the v3 protocol.
    """

    def __init__(self) -> None:
        """
        Initializes the A2ADiscoveryCardsV3.
        """
        pass

    async def fetch_and_parse_cards(self, endpoint_url: str) -> List[AgentCardV3]:
        """
        Asynchronously requests an endpoint to retrieve JSON formatted Agent Cards,
        and parses them into a list of AgentCardV3 objects.

        Args:
            endpoint_url: The URL of the A2A discovery endpoint.

        Returns:
            A list of successfully parsed AgentCardV3 objects.

        Raises:
            httpx.HTTPError: If the HTTP request fails.
        """
        logging.info(f"Fetching Agent Cards from {endpoint_url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint_url)
            response.raise_for_status()

            data = response.json()

            # Assuming the endpoint returns a list of JSON strings or dictionaries.
            parsed_cards: List[AgentCardV3] = []

            if not isinstance(data, list):
                logging.error(f"Expected a list of cards, got {type(data)}")
                return parsed_cards

            for item in data:
                try:
                    if isinstance(item, str):
                        card = AgentCardV3.from_json(item)
                    elif isinstance(item, dict):
                        # Ensure protocol version is v3 or missing
                        card = AgentCardV3(**item)
                    else:
                        continue
                    parsed_cards.append(card)
                    logging.info(f"Successfully parsed AgentCard for agent_id: {card.agent_id}")
                except Exception as e:
                    logging.error(f"Failed to parse Agent Card. Error: {str(e)}")

            return parsed_cards
