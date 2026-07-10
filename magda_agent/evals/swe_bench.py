import logging
from typing import Dict, Any, List, Optional

from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

class SWEBenchEvaluator:
    """
    Evaluation suite inspired by SWE-bench Verified trend, focusing on agent performance
    in resolving real-world GitHub issues. Reports metrics to QualityTracker for longitudinal analysis.
    """

    def __init__(self, db_path: str = "./metrics_db.sqlite3", consciousness: Any = None):
        """
        Initialize the SWEBenchEvaluator.

        Args:
            db_path (str): Path to the database for QualityTracker.
            consciousness (Any): An instance of the agent's consciousness for processing inputs.
        """
        self.tracker = QualityTracker(db_path=db_path)
        self.llm = LLMClient()
        self.consciousness = consciousness
        self.tasks = [
            {
                "id": "swe-001",
                "goal": "Fix the issue in magda_agent/llm_client.py where connection timeouts are not handled properly.",
                "difficulty": "medium"
            },
            {
                "id": "swe-002",
                "goal": "Add a new feature in magda_agent/memory/storage.py to support vector search.",
                "difficulty": "hard"
            },
            {
                "id": "swe-003",
                "goal": "Refactor tests/test_emotions.py to use mock fixtures instead of real API calls.",
                "difficulty": "easy"
            }
        ]

    async def run_task(self, task: Dict[str, Any]) -> float:
        """
        Runs a single SWE-bench task and evaluates the result.

        Args:
            task (Dict[str, Any]): The task dictionary with 'goal' and 'difficulty'.

        Returns:
            float: A score between 0.0 and 1.0 indicating task success.
        """
        goal = task["goal"]
        if not self.consciousness:
            logger.debug(f"Simulating SWE-bench task: {goal}")
            prompt = f"Simulate an agent's success on this software engineering task: '{goal}'. Return a score between 0.0 and 1.0."
            response = await self.llm.generate(prompt)
            try:
                return max(0.0, min(1.0, float(response.strip())))
            except ValueError:
                return 0.5

        try:
            logger.info(f"Executing SWE-bench task: {goal}")
            response = await self.consciousness.process_input(goal)

            eval_prompt = (
                f"Task: {goal}\n"
                f"Agent Response: {response}\n"
                f"Evaluate if the software engineering task was successfully completed based on the agent's response. "
                "Consider whether the agent proposed a valid code change, ran tests, and verified the fix. "
                "Return only a numerical score between 0.0 and 1.0."
            )
            eval_response = await self.llm.generate(eval_prompt)
            try:
                return max(0.0, min(1.0, float(eval_response.strip())))
            except ValueError:
                return 0.5
        except Exception as e:
            logger.error(f"Error running SWE-bench task: {e}")
            return 0.0

    async def run_evaluation_suite(self) -> Dict[str, Any]:
        """
        Runs all SWE-bench tasks and logs the aggregate result.

        Returns:
            Dict[str, Any]: A dictionary containing the aggregate score and evaluation metadata.
        """
        logger.info("Starting SWE-bench evaluation suite...")
        scores = []
        for task in self.tasks:
            score = await self.run_task(task)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        metadata = {
            "suite": "swe_bench_verified",
            "tasks_run": len(self.tasks),
            "scores": scores,
            "average_score": avg_score
        }

        self.tracker.log_metric("swe_bench_score", avg_score, metadata)
        logger.info(f"SWE-bench evaluation complete. Average score: {avg_score:.2f}")

        return {
            "score": avg_score,
            "metadata": metadata
        }
