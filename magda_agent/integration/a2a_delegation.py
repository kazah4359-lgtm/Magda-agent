from typing import Dict, Any
import logging
from magda_agent.integration.a2a_discovery import A2ADiscovery

class A2ADelegator:
    """
    Handles delegating task sub-plans to external agents via A2ADiscovery.
    """
    def __init__(self, discovery: A2ADiscovery):
        """
        Initializes the delegator with the discovery component.
        """
        self.discovery = discovery

    async def delegate_subplan(self, capability: str, plan_context: Dict[str, Any]) -> str:
        """
        Finds an agent capable of executing the requested capability and delegates
        the subplan to it. In a real environment, this would establish an MCP connection.

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
        # Mocking the MCP call and returning a successful result
        return f"Delegated to Agent {target_agent.name}"
