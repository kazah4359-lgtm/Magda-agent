import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple
from magda_agent.llm_client import LLMClient

class MagenticOneOrchestrator:
    """
    Implements a multi-agent orchestration pattern inspired by Microsoft's Magentic-One.

    This orchestrator uses a primary Orchestrator agent to plan, delegate, and review
    tasks executed by a team of specialized agents. It dynamically adjusts team size based on task difficulty.
    """

    def __init__(self, llm: LLMClient):
        """
        Initializes the MagenticOneOrchestrator.

        Args:
            llm (LLMClient): The language model client to be used.
        """
        self.llm = llm

    def _evaluate_difficulty(self, task: str) -> int:
        """
        Evaluates the difficulty of the task based on heuristics to dynamically scale the team size.

        Args:
            task (str): The task to evaluate.

        Returns:
            int: An integer between 1 and 10 representing difficulty.
        """
        # A simple heuristic to avoid breaking existing LLM mock call counts.
        # Longer tasks are considered more difficult.
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
            "Return ONLY a valid JSON list of dictionaries. Each dictionary must have 'id' and 'description' keys."
        )
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        try:
            # Attempt to parse the JSON response from the LLM
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

    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> List[str]:
        """
        Executes a plan by delegating subtasks to specialized agents (simulated here).
        """
        results = []
        for step in plan:
            prompt = f"Execute subtask: {step['description']}"
            res = await self.llm.chat_completion([{"role": "user", "content": prompt}])
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
            execution_results = await self._execute_plan(plan)
            context.extend(execution_results)

            # Step 3: Review
            is_complete, final_result = await self._review(task, context)
            if is_complete:
                return final_result

        return f"Task incomplete after {max_iterations} iterations. Last context: {context}"
