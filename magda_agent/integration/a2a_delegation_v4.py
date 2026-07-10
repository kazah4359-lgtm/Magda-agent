from typing import Any, Dict, List, Optional
import httpx
import logging

from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.integration.a2a_tracing import A2ATracer

logger = logging.getLogger(__name__)

class A2ADelegatorV4:
    """
    Handles peer-to-peer task delegation using Agent Cards for asynchronous discovery.
    """
    def __init__(self, security_context: Optional[A2ASecurityContext] = None) -> None:
        """
        Initializes the delegator.
        """
        self.security_context = security_context or A2ASecurityContext()

    async def delegate_task(self, peer_agent: AgentCardV3, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates a task to a peer agent asynchronously.

        Args:
            peer_agent (AgentCardV3): The agent card representing the peer.
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
            self.security_context.trace_action("delegate_task", {"target_agent": peer_agent.name})

        A2ATracer.record_event("peer_delegation", {"target_agent_id": peer_agent.agent_id})

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=task_payload, headers=headers, timeout=10.0)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to delegate task to {peer_agent.agent_id} at {endpoint}: {e}")
            raise
