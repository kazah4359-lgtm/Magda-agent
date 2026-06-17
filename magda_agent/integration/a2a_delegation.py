from typing import Dict, Any
import logging
from magda_agent.integration.a2a_discovery import A2ADiscovery
from magda_agent.integration.a2a_tracing import A2ATracer
import httpx

class A2ADelegator:
    """
    Handles delegating task sub-plans to external agents via A2ADiscovery.
    """
    def __init__(self, discovery: A2ADiscovery):
        """
        Initializes the delegator with the discovery component.
        """
        self.discovery = discovery
        self.security_context = getattr(discovery, 'security_context', None)


    async def delegate_subplan(self, capability: str, plan_context: Dict[str, Any]) -> str:
        """
        Finds an agent capable of executing the requested capability and delegates
        the subplan to it dynamically over the network using httpx.

        Args:
            capability: The required capability (e.g., 'code_execution').
            plan_context: The task context or sub-plan.

        Returns:
            A result string describing the outcome.
        """
        agents = self.discovery.find_agents_by_capability(capability)
        if not agents:
            logging.warning(f"No agents found for capability: {capability}")
            return "No agent found"

        # Select the first available agent
        target_agent = agents[0]

        logging.info(f"Delegating sub-plan to Agent: {target_agent.name} (ID: {target_agent.agent_id})")

        endpoint = target_agent.endpoints.get("mcp")
        if not endpoint:
            return f"Agent {target_agent.name} missing MCP endpoint"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "execute_subplan",
            "params": {"capability": capability, "context": plan_context}
        }

        headers = {}
        # Inject distributed tracing header
        A2ATracer.inject_headers(headers)

        if self.security_context:
            token = self.security_context.generate_token()
            headers["Authorization"] = f"Bearer {token}"
            self.security_context.trace_action("delegate_subplan", {"capability": capability, "target_agent": target_agent.name})

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=payload, headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                result = data.get("result", {})
                return f"Delegated to Agent {target_agent.name}: {result.get('status', 'Success')}"
        except Exception as e:
            logging.error(f"Failed to delegate to {target_agent.name} at {endpoint}: {e}")
            return f"Delegation to {target_agent.name} failed: {e}"
