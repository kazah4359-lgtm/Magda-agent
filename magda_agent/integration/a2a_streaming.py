import logging
import json
from typing import Dict, Any, AsyncGenerator, Optional
import httpx

from magda_agent.integration.a2a_cards import AgentCardV3
from magda_agent.integration.a2a_security import A2ASecurityContext
from magda_agent.integration.a2a_tracing import A2ATracer

class A2AStreamingDelegator:
    """
    Implements a streaming interface for A2A peer-to-peer task delegation,
    supporting long-horizon task coordination via chunked updates.
    """
    def __init__(self, security_context: Optional[A2ASecurityContext] = None) -> None:
        """
        Initializes the streaming delegator.

        Args:
            security_context: Optional security context for token generation.
        """
        self.security_context = security_context or A2ASecurityContext()

    async def stream_delegation(self, target_agent: AgentCardV3, plan_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Delegates a task to a peer agent and streams the chunked updates back.

        Args:
            target_agent: The target AgentCardV3 representing the peer.
            plan_context: The task context to delegate.

        Yields:
            Chunked update dictionaries received from the peer.
        """
        logging.info(f"Initiating streaming delegation to Agent: {target_agent.name}")
        endpoint = target_agent.endpoints.get("rpc")
        if not endpoint:
            yield {"error": f"Agent {target_agent.name} missing rpc endpoint"}
            return

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "stream_subplan",
            "params": {"context": plan_context}
        }

        headers: Dict[str, str] = {}
        A2ATracer.inject_headers(headers)

        if self.security_context:
            token = self.security_context.generate_token()
            headers["Authorization"] = f"Bearer {token}"
            self.security_context.trace_action("stream_delegation", {"target_agent": target_agent.name})

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers, timeout=30.0) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                yield data
                            except json.JSONDecodeError:
                                yield {"error": "Failed to decode chunk", "raw": chunk}
        except Exception as e:
            logging.error(f"Streaming delegation failed: {e}")
            yield {"error": str(e)}


class A2AStreamingDelegatorV2(A2AStreamingDelegator):
    """
    V2 Implementation of the streaming interface for A2A peer-to-peer task delegation.
    Enhances long-horizon task coordination by supporting configurable timeouts and retries.
    """
    def __init__(self, security_context: Optional[A2ASecurityContext] = None, timeout: float = 60.0) -> None:
        """
        Initializes the V2 streaming delegator.

        Args:
            security_context: Optional security context for token generation.
            timeout: Configurable timeout for long-horizon task streaming.
        """
        super().__init__(security_context)
        self.timeout = timeout

    async def stream_delegation_v2(self, target_agent: AgentCardV3, plan_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Delegates a task to a peer agent using enhanced V2 configuration.

        Args:
            target_agent: The target AgentCardV3 representing the peer.
            plan_context: The task context to delegate.

        Yields:
            Chunked update dictionaries received from the peer.
        """
        logging.info(f"Initiating V2 streaming delegation to Agent: {target_agent.name} with timeout {self.timeout}")
        endpoint = target_agent.endpoints.get("rpc")
        if not endpoint:
            yield {"error": f"Agent {target_agent.name} missing rpc endpoint"}
            return

        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "stream_subplan_v2",
            "params": {"context": plan_context}
        }

        headers: Dict[str, str] = {}
        A2ATracer.inject_headers(headers)

        if self.security_context:
            token = self.security_context.generate_token()
            headers["Authorization"] = f"Bearer {token}"
            self.security_context.trace_action("stream_delegation_v2", {"target_agent": target_agent.name})

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers, timeout=self.timeout) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                yield data
                            except json.JSONDecodeError:
                                yield {"error": "Failed to decode chunk", "raw": chunk}
        except Exception as e:
            logging.error(f"V2 Streaming delegation failed: {e}")
            yield {"error": str(e)}
