import asyncio
import logging
from typing import Dict, Any, Optional, List
from magda_agent.llm_client import LLMClient
from magda_agent.agents.sub_agent import SubAgent

class SubAgentRPCManager:
    """
    OpenClaw Sub-agents RPC v1 Manager
    Implements a manager that orchestrates sub-agents for parallel tasks with RPC and isolated context.
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.active_agents: Dict[str, SubAgent] = {}

    async def spawn_agent(self, agent_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Spawns a new isolated sub-agent and registers it.
        """
        logging.info(f"Spawning RPC SubAgent: {agent_id}")
        system_prompt = "You are an isolated RPC Sub-Agent."
        if context:
            system_prompt += f" Context: {context}"

        # SubAgent uses Git worktrees when use_isolation=True, giving it its own execution sandbox
        agent = SubAgent(llm=self.llm, system_prompt=system_prompt, use_isolation=True)
        self.active_agents[agent_id] = agent

    async def execute_task_rpc(self, agent_id: str, task: str, base_context: str = "") -> Dict[str, Any]:
        """
        Executes a task via an RPC-like interface to the specified sub-agent.
        """
        if agent_id not in self.active_agents:
            return {"status": "error", "error": f"Agent {agent_id} not found."}

        logging.info(f"RPC call to agent {agent_id} for task: {task[:30]}")
        agent = self.active_agents[agent_id]

        try:
            result = await agent.execute(task=task, context=base_context)
            return {"status": "success", "agent_id": agent_id, "result": result}
        except Exception as e:
            logging.error(f"RPC Error for agent {agent_id}: {e}")
            return {"status": "error", "agent_id": agent_id, "error": str(e)}

    async def kill_agent(self, agent_id: str) -> None:
        """
        Terminates the sub-agent session.
        """
        if agent_id in self.active_agents:
            logging.info(f"Killing RPC SubAgent: {agent_id}")
            del self.active_agents[agent_id]
