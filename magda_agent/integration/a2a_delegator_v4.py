import httpx
import logging
from typing import Dict, Any

class A2ADelegatorV4:
    """
    Handles peer-to-peer task delegation using the A2A spec.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    async def delegate_task(self, peer_endpoint: str, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Delegates a task to a peer agent asynchronously.

        Args:
            peer_endpoint: The URL endpoint of the peer agent.
            payload: The task payload to send.
            headers: Optional HTTP headers for the request.

        Returns:
            A dictionary containing the response from the peer.

        Raises:
            httpx.HTTPError: If the HTTP request fails.
        """
        logging.info(f"Delegating task to peer at {peer_endpoint}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    peer_endpoint,
                    json=payload,
                    headers=headers or {},
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logging.error(f"Failed to delegate task to {peer_endpoint}: {e}")
                raise
