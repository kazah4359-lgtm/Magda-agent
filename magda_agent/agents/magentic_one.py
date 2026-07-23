import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from magda_agent.llm_client import LLMClient

class MagenticOneWorker:
    """
    A specialized worker agent in Microsoft's Magentic-One orchestration pattern.
    """
    def __init__(self, name: str, description: str, llm: LLMClient):
        self.name = name
        self.description = description
        self.llm = llm

    async def execute_subtask(self, subtask: str, context: List[str]) -> str:
        """
        Executes a specific subtask using the worker's specialized capabilities.
        """
        prompt = (
            f"You are the specialized agent {self.name} ({self.description}).\n"
            f"Execute this task: {subtask}\n"
            f"Current context: {context}\n"
            "Return the outcome."
        )
        try:
            return await self.llm.chat_completion([{"role": "user", "content": prompt}])
        except Exception as e:
            logging.error(f"Worker {self.name} failed: {e}")
            return f"Worker {self.name} encountered an error: {e}"


class MagenticOneOrchestrator:
    """
    Implements a multi-agent orchestration pattern inspired by Microsoft's Magentic-One.

    This orchestrator uses a primary Orchestrator agent to plan, delegate, and review
    tasks executed by a team of specialized agents. It dynamically adjusts team size based on task difficulty.
    Supports hierarchical delegation where tasks are recursively delegated to sub-teams or sub-agents.
    """

    def __init__(self, llm: LLMClient, workers: Optional[List[MagenticOneWorker]] = None):
        """
        Initializes the MagenticOneOrchestrator.

        Args:
            llm (LLMClient): The language model client to be used.
            workers (Optional[List[MagenticOneWorker]]): A list of available worker agents.
        """
        self.llm = llm
        if workers is None:
            self.workers = [
                MagenticOneWorker("WebSurfer", "Specialized in web browsing, search, and navigation.", llm),
                MagenticOneWorker("FileSurfer", "Specialized in reading, writing, and navigating the filesystem.", llm),
                MagenticOneWorker("Coder", "Specialized in writing code and logic scripts.", llm),
                MagenticOneWorker("Executor", "Specialized in compiling, executing, and testing code.", llm)
            ]
        else:
            self.workers = workers

    def _evaluate_difficulty(self, task: str) -> int:
        """
        Evaluates the difficulty of the task based on heuristics to dynamically scale the team size.

        Args:
            task (str): The task to evaluate.

        Returns:
            int: An integer between 1 and 10 representing difficulty.
        """
        length = len(task)
        if length < 20:
            return 2
        elif length < 50:
            return 5
        elif length < 100:
            return 8
        else:
            return 10

    def _calculate_team_size(self, difficulty: int) -> int:
        """
        Calculates the appropriate team size based on task difficulty.

        Args:
            difficulty (int): The evaluated task difficulty (1-10).

        Returns:
            int: The calculated team size (1-5).
        """
        if difficulty <= 2:
            return 1
        elif difficulty <= 4:
            return 2
        elif difficulty <= 6:
            return 3
        elif difficulty <= 8:
            return 4
        else:
            return 5

    async def _plan(self, task: str, context: List[str], team_size: int = None) -> List[Dict[str, Any]]:
        """
        Creates a plan by generating subtasks based on the main task, current context, and team size.
        """
        size_prompt = f" Generate EXACTLY {team_size} subtasks." if team_size else ""
        prompt = (
            f"Plan task: {task}. Context: {context}.{size_prompt} "
            "Return ONLY a valid JSON list of dictionaries. Each dictionary must have 'id' and 'description' keys. "
            "Optionally, a dictionary can also contain a nested 'subtasks' key containing further subtask lists for hierarchical delegation."
        )
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3].strip()

            parsed_plan = json.loads(clean_response)
            if isinstance(parsed_plan, list) and len(parsed_plan) > 0 and 'id' in parsed_plan[0] and 'description' in parsed_plan[0]:
                return parsed_plan[:team_size] if team_size else parsed_plan
        except Exception as e:
            logging.error(f"Failed to parse plan JSON: {e}. Raw response: {response}")

        # Fallback plan if parsing fails
        num_tasks = team_size if team_size else 1
        return [{"id": f"fallback_subtask_{i}", "description": f"Attempt subtask {i} for {task}"} for i in range(1, num_tasks + 1)]

    async def _execute_step(self, step: Dict[str, Any], context: List[str]) -> str:
        """
        Executes a single step in the plan, routing to specialized workers if possible, or falling back
        to orchestrator execution to ensure backward compatibility.
        """
        description = step.get('description', '')
        worker_name = step.get('worker')

        # Check if the step itself has nested subtasks for hierarchical delegation
        if "subtasks" in step and isinstance(step["subtasks"], list) and len(step["subtasks"]) > 0:
            logging.info(f"Hierarchical delegation triggered for subtasks of: {description}")
            sub_results = []
            for sub_step in step["subtasks"]:
                sub_res = await self._execute_step(sub_step, context)
                sub_results.append(sub_res)
            return f"Hierarchical Delegation Result for '{description}': {sub_results}"

        if worker_name:
            worker = next((w for w in self.workers if w.name.lower() == worker_name.lower()), None)
            if worker:
                return await worker.execute_subtask(description, context)

        # Fallback/default prompt execution pattern (ensures mock compatibility)
        prompt = f"Execute subtask: {description}"
        return await self.llm.chat_completion([{"role": "user", "content": prompt}])

    async def _execute_plan(self, plan: List[Dict[str, Any]], context: List[str] = None) -> List[str]:
        """
        Executes a plan by delegating subtasks (potentially hierarchically).
        """
        if context is None:
            context = []
        results = []
        for step in plan:
            res = await self._execute_step(step, context)
            results.append(res)
        return results

    async def _review(self, task: str, context: List[str]) -> Tuple[bool, str]:
        """
        Reviews the current context to determine if the main task is complete.
        """
        prompt = f"Review task: {task}. Context: {context}. Is complete? Return YES or NO, then the result."
        res = await self.llm.chat_completion([{"role": "user", "content": prompt}])

        is_complete = "YES" in res.upper()
        return is_complete, res

    async def orchestrate(self, task: str, max_iterations: int = 3) -> str:
        """
        Orchestrates the execution of a complex task by planning, delegating, and reviewing.

        Args:
            task (str): The main task to accomplish.
            max_iterations (int): Maximum number of plan-execute-review loops.

        Returns:
            str: The final result of the orchestration process.
        """
        context: List[str] = []

        difficulty = self._evaluate_difficulty(task)
        team_size = self._calculate_team_size(difficulty)

        for iteration in range(max_iterations):
            # Step 1: Planning
            plan = await self._plan(task, context, team_size)

            # Step 2: Delegation and Execution
            execution_results = await self._execute_plan(plan, context)
            context.extend(execution_results)

            # Step 3: Review
            is_complete, final_result = await self._review(task, context)
            if is_complete:
                return final_result

        return f"Task incomplete after {max_iterations} iterations. Last context: {context}"
