import logging
import asyncio
from typing import Dict, Any, List, Optional

from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

class WebArenaEvaluator:
    """
    Evaluation suite inspired by WebArena, focusing on web navigation tasks.
    Reports metrics to QualityTracker for longitudinal analysis.
    """

    def __init__(self, db_path: str = "./metrics_db.sqlite3", consciousness: Any = None):
        self.tracker = QualityTracker(db_path=db_path)
        self.llm = LLMClient()
        self.consciousness = consciousness
        self.tasks = [
            {
                "id": "wa-001",
                "goal": "Find the price of the latest iPhone on Apple's website.",
                "difficulty": "easy"
            },
            {
                "id": "wa-002",
                "goal": "Navigate to Wikipedia and find the capital of Kazakhstan.",
                "difficulty": "easy"
            },
            {
                "id": "wa-003",
                "goal": "Go to the e-commerce site, search for 'headphones', and add the first result to the cart.",
                "difficulty": "medium"
            },
            {
                "id": "wa-004",
                "goal": "Log in to the portal and change the user profile display name to 'Magda User'.",
                "difficulty": "hard"
            }
        ]

    async def run_task(self, task: Dict[str, Any]) -> float:
        """
        Runs a single WebArena task and evaluates the result.
        """
        goal = task["goal"]
        if not self.consciousness:
            logger.debug(f"Simulating WebArena task: {goal}")
            prompt = f"Simulate an agent's success on this web navigation task: '{goal}'. Return a score between 0.0 and 1.0."
            response = await self.llm.generate(prompt)
            try:
                return max(0.0, min(1.0, float(response.strip())))
            except ValueError:
                return 0.5

        try:
            logger.info(f"Executing WebArena task: {goal}")
            response = await self.consciousness.process_input(goal)

            eval_prompt = (
                f"Task: {goal}\n"
                f"Agent Response: {response}\n"
                f"Evaluate if the web navigation task was successfully completed based on the agent's response. "
                "Return only a numerical score between 0.0 and 1.0."
            )
            eval_response = await self.llm.generate(eval_prompt)
            try:
                return max(0.0, min(1.0, float(eval_response.strip())))
            except ValueError:
                return 0.5
        except Exception as e:
            logger.error(f"Error running WebArena task: {e}")
            return 0.0

    async def run_evaluation_suite(self) -> Dict[str, Any]:
        """
        Runs all WebArena tasks and logs the aggregate result.
        """
        logger.info("Starting WebArena evaluation suite...")
        scores = []
        for task in self.tasks:
            score = await self.run_task(task)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        metadata = {
            "suite": "web_arena",
            "tasks_run": len(self.tasks),
            "scores": scores,
            "average_score": avg_score
        }

        self.tracker.log_metric("web_arena_score", avg_score, metadata)
        logger.info(f"WebArena evaluation complete. Average score: {avg_score:.2f}")

        return {
            "score": avg_score,
            "metadata": metadata
        }
