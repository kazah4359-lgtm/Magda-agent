from typing import Dict, Any, List, Optional
import json
import logging
import httpx
from magda_agent.integration.a2a_discovery import AgentCard

class A2ADiscoveryAgent:
    """
    Agent responsible for discovering other agents and delegating tasks.
    Supports compatibility with external frameworks like LangGraph and CrewAI.
    """

    def __init__(self, agent_id: str, name: str, capabilities: List[str]):
        """
        Initializes the A2ADiscoveryAgent.

        Args:
            agent_id: Unique identifier for this agent.
            name: Human-readable name of the agent.
            capabilities: List of capabilities this agent supports.
        """
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.known_agents: Dict[str, AgentCard] = {}

    def generate_my_card(self, endpoints: Dict[str, str]) -> AgentCard:
        """
        Generates an Agent Card for this agent.

        Args:
            endpoints: Dictionary of endpoints (e.g., {'rpc': 'http://...'})

        Returns:
            An AgentCard object representing this agent.
        """
        return AgentCard(
            agent_id=self.agent_id,
            name=self.name,
            description=f"Magda Agent: {self.name}",
            capabilities=self.capabilities,
            endpoints=endpoints
        )

    def parse_card(self, card_json: str) -> Optional[AgentCard]:
        """
        Parses an Agent Card from JSON and adds it to known agents.

        Args:
            card_json: JSON string representing an Agent Card.

        Returns:
            The parsed AgentCard if successful, else None.
        """
        try:
            card = AgentCard.from_json(card_json)
            self.known_agents[card.agent_id] = card
            return card
        except Exception as e:
            logging.error(f"Failed to parse agent card: {e}")
            return None

    async def delegate_to_external(self, target_agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates a task to an external agent (e.g., LangGraph/CrewAI).
        Mocks the actual network call if no real endpoint is available.

        Args:
            target_agent_id: The ID of the agent to delegate to.
            task: The task details to delegate.

        Returns:
            A dictionary containing the result of the delegation.
        """
        agent = self.known_agents.get(target_agent_id)
        if not agent:
            return {"status": "error", "message": f"Agent {target_agent_id} not found"}

        endpoint = agent.endpoints.get("rpc") or agent.endpoints.get("a2a")
        if not endpoint:
            # Mock behavior for external agents without real endpoints in test env
            logging.info(f"Mocking delegation to external agent {agent.name} ({agent.agent_id})")
            return {
                "status": "success",
                "message": f"Mocked delegation to {agent.name}",
                "delegated_task": task,
                "framework": "LangGraph" if "langgraph" in agent.description.lower() else "CrewAI"
            }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=task, timeout=10.0)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logging.error(f"Failed to delegate to {agent.name}: {e}")
            return {"status": "error", "message": str(e)}

    def find_peers_by_capability(self, capability: str) -> List[AgentCard]:
        """
        Finds known agents that support a specific capability.

        Args:
            capability: The capability to search for.

        Returns:
            A list of AgentCard objects supporting the capability.
        """
        return [agent for agent in self.known_agents.values() if capability in agent.capabilities]
