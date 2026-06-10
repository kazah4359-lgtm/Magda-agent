import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, Response
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

    async def execute_parallel_rpc(self, agent_ids: List[str], tasks: List[str], base_context: str = "") -> List[Dict[str, Any]]:
        """
        Executes multiple tasks in parallel across multiple sub-agents.
        """
        if len(agent_ids) != len(tasks):
            raise ValueError("agent_ids and tasks must have the same length.")

        coros = [self.execute_task_rpc(aid, task, base_context) for aid, task in zip(agent_ids, tasks)]
        return list(await asyncio.gather(*coros))

    async def kill_agent(self, agent_id: str) -> None:
        """
        Terminates the sub-agent session.
        """
        if agent_id in self.active_agents:
            logging.info(f"Killing RPC SubAgent: {agent_id}")
            del self.active_agents[agent_id]


class SubAgentRPCServer:
    """
    A JSON-RPC 2.0 Server interface for Sub-agent RPC management.
    """
    def __init__(self, manager: SubAgentRPCManager) -> None:
        self.manager = manager
        self.app = FastAPI(title="Sub-agent RPC Server")

        @self.app.post("/rpc")
        async def handle_rpc(request: Request) -> Response:
            return await self.handle_request(request)

    async def handle_request(self, request: Request) -> Response:
        try:
            data = await request.json()
        except Exception:
            return self._error_response(-32700, "Parse error")

        if not isinstance(data, dict):
            return self._error_response(-32600, "Invalid Request")

        req_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {})

        if data.get("jsonrpc") != "2.0" or not method:
            return self._error_response(-32600, "Invalid Request", req_id)

        try:
            if method == "spawn_agent":
                agent_id = params.get("agent_id")
                context = params.get("context")
                if not agent_id:
                    return self._error_response(-32602, "Invalid params: agent_id required", req_id)
                await self.manager.spawn_agent(agent_id, context)
                return self._result_response({"status": "spawned", "agent_id": agent_id}, req_id)

            elif method == "execute_task":
                agent_id = params.get("agent_id")
                task = params.get("task")
                context = params.get("context", "")
                if not agent_id or not task:
                    return self._error_response(-32602, "Invalid params: agent_id and task required", req_id)
                result = await self.manager.execute_task_rpc(agent_id, task, context)
                return self._result_response(result, req_id)

            elif method == "kill_agent":
                agent_id = params.get("agent_id")
                if not agent_id:
                    return self._error_response(-32602, "Invalid params: agent_id required", req_id)
                await self.manager.kill_agent(agent_id)
                return self._result_response({"status": "killed", "agent_id": agent_id}, req_id)

            else:
                return self._error_response(-32601, "Method not found", req_id)

        except Exception as e:
            return self._error_response(-32000, str(e), req_id)

    def _error_response(self, code: int, message: str, req_id: Any = None) -> Response:
        return Response(content=json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id
        }), media_type="application/json")

    def _result_response(self, result: Any, req_id: Any) -> Response:
        return Response(content=json.dumps({
            "jsonrpc": "2.0",
            "result": result,
            "id": req_id
        }), media_type="application/json")
