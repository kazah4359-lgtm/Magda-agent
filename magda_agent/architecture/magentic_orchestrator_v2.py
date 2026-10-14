import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from magda_agent.llm_client import LLMClient


class MagenticBlackboardV2:
    """
    Thread-safe and async-safe global blackboard state shared across sub-agents in Magentic-One pattern.
    Protects state mutations with an asyncio.Lock to prevent race conditions during concurrent updates.
    """

    def __init__(self, initial_facts: Optional[List[str]] = None) -> None:
        self._lock = asyncio.Lock()
        self._facts: List[str] = list(initial_facts) if initial_facts else []
        self._state: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []

    async def add_fact(self, fact: str, author: str = "") -> None:
        """
        Safely appends a new fact to the shared blackboard.
        """
        async with self._lock:
            self._facts.append(fact)
            self._history.append({"action": "add_fact", "author": author, "content": fact})

    async def get_facts(self) -> List[str]:
        """
        Safely retrieves a copy of all current facts on the blackboard.
        """
        async with self._lock:
            return list(self._facts)

    async def set_value(self, key: str, value: Any, author: str = "") -> None:
        """
        Safely sets a key-value entry in the global blackboard state.
        """
        async with self._lock:
            self._state[key] = value
            self._history.append({"action": "set_value", "author": author, "key": key, "value": value})

    async def get_value(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieves a key value from the blackboard state.
        """
        async with self._lock:
            return self._state.get(key, default)

    async def snapshot(self) -> Dict[str, Any]:
        """
        Safely takes a complete snapshot of facts, state entries, and history.
        """
        async with self._lock:
            return {
                "facts": list(self._facts),
                "state": dict(self._state),
                "history": list(self._history),
            }


class MagenticWorkerV2:
    """
    A specialized worker sub-agent in Magentic-One V2.
    Reads shared context from MagenticBlackboardV2 and safely posts results back to it.
    """

    def __init__(self, name: str, role: str, llm: LLMClient) -> None:
        self.name = name
        self.role = role
        self.llm = llm

    async def execute_subtask(self, subtask: str, blackboard: MagenticBlackboardV2) -> str:
        """
        Executes a subtask using LLMClient with the blackboard facts as context,
        and posts the result back to the shared blackboard.
        """
        facts = await blackboard.get_facts()
        prompt = (
            f"You are the specialized sub-agent {self.name} ({self.role}).\n"
            f"Execute this subtask: {subtask}\n"
            f"Shared Blackboard Facts: {facts}\n"
            "Return the outcome."
        )
        try:
            res = await self.llm.chat_completion([{"role": "user", "content": prompt}])
            clean_res = res.strip()
            await blackboard.add_fact(f"[{self.name}] {clean_res}", author=self.name)
            return clean_res
        except Exception as e:
            logging.error(f"Worker {self.name} failed: {e}")
            err_msg = f"Worker {self.name} error: {e}"
            await blackboard.add_fact(f"[{self.name}] {err_msg}", author=self.name)
            return err_msg


class MagenticOrchestratorV2:
    """
    Central Orchestrator based on Microsoft's Magentic-One pattern.
    Maintains a central MagenticBlackboardV2 shared safely across specialized sub-agents.
    """

    def __init__(
        self,
        llm: LLMClient,
        workers: Optional[List[MagenticWorkerV2]] = None,
        blackboard: Optional[MagenticBlackboardV2] = None,
    ) -> None:
        self.llm = llm
        self.blackboard = blackboard if blackboard is not None else MagenticBlackboardV2()
        if workers is None:
            self.workers = [
                MagenticWorkerV2("WebSurfer", "Specialized in web browsing, search, and navigation.", llm),
                MagenticWorkerV2("FileSurfer", "Specialized in reading, writing, and navigating the filesystem.", llm),
                MagenticWorkerV2("Coder", "Specialized in writing code and logic scripts.", llm),
                MagenticWorkerV2("Executor", "Specialized in compiling, executing, and testing code.", llm),
            ]
        else:
            self.workers = workers
        self._current_worker_index = 0

    def _get_next_worker(self) -> MagenticWorkerV2:
        """
        Selects the next worker in round-robin fashion.
        """
        if not self.workers:
            raise ValueError("No workers available")
        worker = self.workers[self._current_worker_index]
        self._current_worker_index = (self._current_worker_index + 1) % len(self.workers)
        return worker

    async def _plan(self, task: str, team_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generates plan steps using the LLM and the current blackboard state.
        """
        facts = await self.blackboard.get_facts()
        size_prompt = f" Generate EXACTLY {team_size} subtasks." if team_size else ""
        prompt = (
            f"Plan task: {task}. Blackboard Facts: {facts}.{size_prompt} "
            "Return ONLY a valid JSON list of dictionaries. Each dictionary must have 'id' and 'description' keys. "
            "Optionally, a dictionary can contain 'worker' (e.g. Coder, WebSurfer, FileSurfer, Executor)."
        )
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3].strip()

            parsed_plan = json.loads(clean_response)
            if (
                isinstance(parsed_plan, list)
                and len(parsed_plan) > 0
                and "id" in parsed_plan[0]
                and "description" in parsed_plan[0]
            ):
                return parsed_plan[:team_size] if team_size else parsed_plan
        except Exception as e:
            logging.error(f"Failed to parse plan JSON: {e}. Raw response: {response}")

        num_tasks = team_size if team_size else 1
        return [
            {"id": f"fallback_subtask_{i}", "description": f"Attempt subtask {i} for {task}"}
            for i in range(1, num_tasks + 1)
        ]

    async def _execute_step(self, step: Dict[str, Any]) -> str:
        """
        Dispatches a step to a specific or round-robin worker sub-agent.
        """
        description = step.get("description", "")
        worker_name = step.get("worker")

        worker = None
        if worker_name:
            worker = next((w for w in self.workers if w.name.lower() == worker_name.lower()), None)

        if not worker and self.workers:
            worker = self._get_next_worker()

        if worker:
            return await worker.execute_subtask(description, self.blackboard)

        # Direct prompt fallback if no worker
        facts = await self.blackboard.get_facts()
        prompt = f"Execute subtask: {description}. Context facts: {facts}"
        res = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        await self.blackboard.add_fact(f"[Orchestrator] {res}", author="Orchestrator")
        return res

    async def _execute_plan(self, plan: List[Dict[str, Any]], parallel: bool = False) -> List[str]:
        """
        Executes all steps in a plan either sequentially or concurrently via asyncio.gather.
        All worker updates to the blackboard are safe against race conditions.
        """
        if parallel:
            tasks = [self._execute_step(step) for step in plan]
            results = await asyncio.gather(*tasks)
            return list(results)
        else:
            results = []
            for step in plan:
                res = await self._execute_step(step)
                results.append(res)
            return results

    async def _review(self, task: str) -> Tuple[bool, str]:
        """
        Reviews current blackboard state to determine if main task is accomplished.
        """
        facts = await self.blackboard.get_facts()
        prompt = (
            f"Review main task: {task}.\n"
            f"Current Blackboard Facts: {facts}.\n"
            "Is complete? Return YES or NO, then the final result summary."
        )
        res = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        is_complete = "YES" in res.upper()
        return is_complete, res

    async def orchestrate(self, task: str, max_iterations: int = 3, parallel: bool = False) -> str:
        """
        Main orchestration loop: plan, delegate to sub-agents, and review against the blackboard.

        Args:
            task (str): The primary goal or task.
            max_iterations (int): Limit on plan-execute-review loops.
            parallel (bool): Whether to execute subtasks concurrently.

        Returns:
            str: Final result outcome.
        """
        for iteration in range(max_iterations):
            plan = await self._plan(task)
            await self._execute_plan(plan, parallel=parallel)
            is_complete, result = await self._review(task)
            if is_complete:
                return result

        facts = await self.blackboard.get_facts()
        return f"Task incomplete after {max_iterations} iterations. Blackboard facts: {facts}"
