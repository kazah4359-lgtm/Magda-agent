import asyncio
import json
import logging
from typing import List, Dict, Any
from magda_agent.llm_client import LLMClient

class MagenticOneOrchestrator:
    """
    Implements a multi-agent orchestration pattern inspired by Microsoft's Magentic-One.

    This orchestrator uses a primary Orchestrator agent to plan, delegate, and review
    tasks executed by a team of specialized agents.
    """

    def __init__(self, llm: LLMClient):
        """
        Initializes the MagenticOneOrchestrator.

        Args:
            llm (LLMClient): The language model client to be used.
        """
        self.llm = llm

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

        for iteration in range(max_iterations):
            # Step 1: Planning
            plan = await self._plan(task, context)

            # Step 2: Delegation and Execution
            execution_results = await self._execute_plan(plan)
            context.extend(execution_results)

            # Step 3: Review
            is_complete, final_result = await self._review(task, context)
            if is_complete:
                return final_result

        return f"Task incomplete after {max_iterations} iterations. Last context: {context}"

    async def _plan(self, task: str, context: List[str]) -> List[Dict[str, Any]]:
        """
        Creates a plan by generating subtasks based on the main task and current context.
        """
        prompt = (
            f"Plan task: {task}. Context: {context}. "
            "Return ONLY a valid JSON list of dictionaries. Each dictionary must have 'id' and 'description' keys."
        )
        response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
        try:
            # Attempt to parse the JSON response from the LLM
            # Sometimes LLMs wrap JSON in markdown blocks
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3].strip()

            parsed_plan = json.loads(clean_response)
            if isinstance(parsed_plan, list) and len(parsed_plan) > 0 and 'id' in parsed_plan[0] and 'description' in parsed_plan[0]:
                return parsed_plan
        except Exception as e:
            logging.error(f"Failed to parse plan JSON: {e}. Raw response: {response}")

        # Fallback plan if parsing fails
        return [{"id": "fallback_subtask", "description": f"Attempt subtask for {task}"}]

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

    async def _review(self, task: str, context: List[str]) -> tuple[bool, str]:
        """
        Reviews the current context to determine if the main task is complete.
        """
        prompt = f"Review task: {task}. Context: {context}. Is complete? Return YES or NO, then the result."
        res = await self.llm.chat_completion([{"role": "user", "content": prompt}])

        is_complete = "YES" in res.upper()
        return is_complete, res
