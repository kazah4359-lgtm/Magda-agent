import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from magda_agent.llm_client import LLMClient

class Worker:
    """
    A specialized agent worker in the Magentic-One orchestration pattern.
    """

    def __init__(self, name: str, description: str, llm: Optional[LLMClient] = None):
        """
        Initializes the Worker.

        Args:
            name (str): The identifier for this worker.
            description (str): Description of the worker's capabilities.
            llm (Optional[LLMClient]): The language model client for the worker.
        """
        self.name = name
        self.description = description
        self.llm = llm

    async def execute_task(self, task: str) -> str:
        """
        Executes a given task.

        Args:
            task (str): The task to execute.

        Returns:
            str: The result of the execution.
        """
        if self.llm:
            prompt = f"You are {self.name}. Your role: {self.description}. Execute this task: {task}"
            try:
                response = await self.llm.chat_completion([{"role": "user", "content": prompt}])
                return response
            except Exception as e:
                logging.error(f"Worker {self.name} failed to execute task: {e}")
                return f"Worker {self.name} encountered an error: {e}"
        else:
            return f"Worker {self.name} executed task: {task} (No LLM configured)"

class Director:
    """
    The Director agent in the Magentic-One orchestration pattern.
    Delegates tasks to a team of Workers.
    """

    def __init__(self, llm: LLMClient, workers: List[Worker]):
        """
        Initializes the Director.

        Args:
            llm (LLMClient): The language model client for the director.
            workers (List[Worker]): A list of available worker agents.
        """
        self.llm = llm
        self.workers = workers

    async def delegate(self, task: str) -> str:
        """
        Analyzes a task, breaks it down, and delegates it to appropriate workers.

        Args:
            task (str): The main task to accomplish.

        Returns:
            str: The final aggregated result.
        """
        worker_profiles = "\n".join([f"- {w.name}: {w.description}" for w in self.workers])

        prompt = (
            f"You are the Director. Main task: {task}\n"
            f"Available workers:\n{worker_profiles}\n\n"
            "Break down the main task and assign subtasks to the most appropriate workers. "
            "Return ONLY a valid JSON list of dictionaries. Each dictionary must have 'worker_name' and 'subtask' keys."
        )

        try:
            response = await self.llm.chat_completion([{"role": "user", "content": prompt}])

            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3].strip()

            assignments = json.loads(clean_response)
        except Exception as e:
            logging.error(f"Director failed to parse assignments JSON: {e}. Raw response: {response if 'response' in locals() else 'None'}")
            return f"Delegation failed: Could not create assignments."

        results = []

        # Parallel execution of subtasks
        tasks = []
        for assignment in assignments:
            worker_name = assignment.get('worker_name')
            subtask = assignment.get('subtask')

            worker = next((w for w in self.workers if w.name == worker_name), None)

            if worker:
                tasks.append(self._run_worker_task(worker, subtask))
            else:
                results.append(f"Worker {worker_name} not found for subtask: {subtask}")

        if tasks:
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            for res in completed_tasks:
                if isinstance(res, Exception):
                    results.append(f"Error during execution: {res}")
                else:
                    results.append(res)

        # Final synthesis
        synthesis_prompt = (
            f"Main task: {task}\n"
            f"Subtask results:\n" + "\n".join(results) + "\n\n"
            "Synthesize these results into a final coherent response to the main task."
        )

        try:
            final_response = await self.llm.chat_completion([{"role": "user", "content": synthesis_prompt}])
            return final_response
        except Exception as e:
            logging.error(f"Director failed to synthesize results: {e}")
            return f"Synthesis failed. Raw results: {results}"

    async def _run_worker_task(self, worker: Worker, subtask: str) -> str:
        """Helper method to execute a task on a worker and format the result."""
        res = await worker.execute_task(subtask)
        return f"[{worker.name}]: {res}"
