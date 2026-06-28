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




    def split_plan(self, plan: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Extracts sub-plans while preserving chronological order.
        Groups contiguous delegation steps for the same capability into sub-plans.
        Enhancements for v8 include strict fallback handling if capability is missing.

        Args:
            plan: The full execution plan.

        Returns:
            A list of sub-plans (where each sub-plan is a dict containing capability and steps).
        """
        sub_plans: list[Dict[str, Any]] = []
        current_capability: str | None = None
        current_steps: list[Dict[str, Any]] = []

        for step in plan:
            if step.get("skill") == "delegate_to_agent":
                kwargs: Dict[str, Any] = step.get("skill_kwargs") or {}
                capability: str | None = kwargs.get("capability")
                if capability:
                    if capability == current_capability:
                        current_steps.append(step)
                    else:
                        if current_capability is not None:
                            sub_plans.append({"capability": current_capability, "steps": current_steps})
                        current_capability = capability
                        current_steps = [step]
                else:
                    logging.warning(f"Delegation step {step.get('id', 'unknown')} is missing 'capability'. Ignoring for sub-plan.")
                    if current_capability is not None:
                        sub_plans.append({"capability": current_capability, "steps": current_steps})
                        current_capability = None
                        current_steps = []
            else:
                if current_capability is not None:
                    sub_plans.append({"capability": current_capability, "steps": current_steps})
                    current_capability = None
                    current_steps = []

        if current_capability is not None:
            sub_plans.append({"capability": current_capability, "steps": current_steps})

        return sub_plans

    async def delegate_to_peer(self, target_agent: 'AgentCard', plan_context: Dict[str, Any]) -> str:
        """
        Delegates a subplan directly to a specific peer agent using its AgentCard.

        Args:
            target_agent: The target AgentCard representing the peer.
            plan_context: The task context or sub-plan to delegate.

        Returns:
            A result string describing the outcome.
        """
        logging.info(f"Delegating directly to Peer Agent: {target_agent.name} (ID: {target_agent.agent_id})")

        endpoint = target_agent.endpoints.get("mcp")
        if not endpoint:
            return f"Agent {target_agent.name} missing MCP endpoint"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "execute_subplan",
            "params": {"context": plan_context}
        }

        headers = {}
        # Inject distributed tracing header
        A2ATracer.inject_headers(headers)

        # Record explicit peer-to-peer delegation trace
        A2ATracer.record_event("peer_delegation", {"target_agent_id": target_agent.agent_id})

        if self.security_context:
            token = self.security_context.generate_token()
            headers["Authorization"] = f"Bearer {token}"
            self.security_context.trace_action("delegate_to_peer", {"target_agent": target_agent.name})

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=payload, headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                result = data.get("result", {})
                return f"Delegated to Peer Agent {target_agent.name}: {result.get('status', 'Success')}"
        except Exception as e:
            logging.error(f"Failed to delegate to peer {target_agent.name} at {endpoint}: {e}")
            return f"Delegation to peer {target_agent.name} failed: {e}"

    async def execute_plan(self, plan: list[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extracts sub-plans chronologically and delegates them sequentially.

        Args:
            plan: The full execution plan.

        Returns:
            A dictionary mapping step IDs to their delegation result.
        """
        results = {}
        sub_plans = self.split_plan(plan)

        for sub_plan in sub_plans:
            capability = sub_plan["capability"]
            for step in sub_plan["steps"]:
                step_id = step.get("id")
                result = await self.delegate_subplan(capability, step)
                if step_id:
                    results[step_id] = result

        return results

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
