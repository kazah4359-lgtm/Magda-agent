from typing import Dict, List, Optional
import httpx
import logging

logger = logging.getLogger(__name__)

class A2ADiscoveryService:
    """
    A2A Agent Discovery Service.
    Inspired by the A2A trend: Build a discovery service to locate and read Agent Cards of peers in the network.
    """
    def __init__(self, endpoint: str = "http://localhost:8000/discovery"):
        self.endpoint = endpoint
        self.peers: Dict[str, Dict] = {}

    async def discover_peers(self) -> List[Dict]:
        """
        Discovers peers by fetching their Agent Cards from the discovery endpoint.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.endpoint}/cards")
                response.raise_for_status()
                cards = response.json()
                for card in cards:
                    if "id" in card:
                        self.peers[card["id"]] = card
                return cards
        except Exception as e:
            logger.error(f"Failed to discover peers: {e}")
            return []

    def get_peer_card(self, peer_id: str) -> Optional[Dict]:
        """
        Retrieves a cached Agent Card for a specific peer.
        """
        return self.peers.get(peer_id)
