import json
import logging
from typing import Dict, Any, Optional
import httpx
from magda_agent.integration.a2a_discovery import AgentCard

logger = logging.getLogger(__name__)

class A2AStatusBroadcaster:
    """
    Broadcasts the agent's current availability and task load using A2A Agent Cards.
    Inspired by A2A peer discovery and Multi-Agent Orchestration trends.
    """

    def __init__(self, agent_card: AgentCard, broadcast_endpoint: str):
        """
        Initializes the broadcaster.

        Args:
            agent_card: The AgentCard representing this agent.
            broadcast_endpoint: The URL endpoint to broadcast the status to.
        """
        self.agent_card = agent_card
        self.broadcast_endpoint = broadcast_endpoint

    def generate_status_payload(self, is_available: bool, active_tasks: int) -> Dict[str, Any]:
        """
        Generates the payload containing the Agent Card and current status.

        Args:
            is_available: Whether the agent is currently available to take new tasks.
            active_tasks: The number of active tasks the agent is currently handling.

        Returns:
            A dictionary containing the serialized agent card and status info.
        """
        payload = {
            "agent_card": json.loads(self.agent_card.to_json()),
            "status": {
                "is_available": is_available,
                "active_tasks": active_tasks
            }
        }
        return payload

    async def broadcast_status(self, is_available: bool, active_tasks: int) -> bool:
        """
        Asynchronously broadcasts the agent's status to the configured endpoint.
        Handles network errors resiliently.

        Args:
            is_available: Whether the agent is currently available.
            active_tasks: The number of active tasks.

        Returns:
            True if the broadcast was successful, False otherwise.
        """
        payload = self.generate_status_payload(is_available, active_tasks)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.broadcast_endpoint,
                    json=payload,
                    timeout=5.0
                )
                response.raise_for_status()
                logger.info(f"Successfully broadcasted status to {self.broadcast_endpoint}")
                return True
        except httpx.RequestError as e:
            logger.error(f"Network error while broadcasting status: {e}")
            return False
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} while broadcasting status: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while broadcasting status: {e}")
            return False
