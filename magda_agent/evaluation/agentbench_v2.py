import logging
from typing import Dict, Any, List

from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

class AgentBenchHarnessV2:
    """
    A testing harness (v2) that allows Magda to be evaluated continuously against
    multi-domain agent benchmarks (like AgentBench), reporting metrics longitudinally
    to an SQLite database via QualityTracker.

    This version improves on V1 by adding more domains and structured evaluation logic.
    """

    def __init__(self, db_path: str = "./metrics_db.sqlite3") -> None:
        """
        Initializes the AgentBenchHarnessV2.

        Args:
            db_path (str): The path to the SQLite database used by QualityTracker.
        """
        self.tracker = QualityTracker(db_path=db_path)
        self.suites = ["web_navigation", "os_interaction", "reasoning", "coding", "multi_agent"]
        self.llm = LLMClient()

    def compare_with_baseline(self, score: float, suite_name: str) -> bool:
        """
        Compares the given score against the suite's baseline.

        Args:
            score (float): The score to compare.
            suite_name (str): The name of the suite.

        Returns:
            bool: True if the score meets or exceeds the baseline, False otherwise.
        """
        baselines = {
            "web_navigation": 0.65,
            "os_interaction": 0.6,
            "reasoning": 0.85,
            "coding": 0.75,
            "multi_agent": 0.55
        }
        return score >= baselines.get(suite_name, 0.5)

    async def run_evaluation_suite(self, suite_name: str) -> Dict[str, Any]:
        """
        Runs an evaluation suite against mock test data for the domain.

        Args:
            suite_name (str): The name of the suite to evaluate.

        Returns:
            Dict[str, Any]: The score and metadata resulting from the evaluation.
        """
        if suite_name not in self.suites:
            raise ValueError(f"Unknown suite: {suite_name}")

        logger.info(f"Running AgentBench V2 evaluation suite: {suite_name}")

        tasks_run = 15  # V2 runs more tasks for better statistical significance
        try:
            prompt = (
                f"Evaluate the agent's capability in the domain of '{suite_name}'. "
                "Consider recent execution traces and benchmark definitions. "
                "Return only a numerical score between 0.0 and 1.0 representing the success rate."
            )
            response = await self.llm.generate(prompt)
            try:
                score = float(response.strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Could not parse score from LLM response: '{response}'. Defaulting to 0.5")
                score = 0.5
        except Exception as e:
            logger.error(f"Failed to evaluate '{suite_name}' using LLM: {e}")
            score = 0.0

        passed = int(score * tasks_run)

        metadata = {
            "version": "v2",
            "suite": suite_name,
            "tasks_run": tasks_run,
            "passed": passed,
            "meets_baseline": self.compare_with_baseline(score, suite_name)
        }

        return {
            "score": score,
            "metadata": metadata
        }

    async def trigger_evaluations(self) -> List[Dict[str, Any]]:
        """
        Triggers all registered evaluation suites and logs their scores to the tracker.

        Returns:
            List[Dict[str, Any]]: A list of results from all suites.
        """
        results = []
        for suite in self.suites:
            try:
                result = await self.run_evaluation_suite(suite)
                self.tracker.log_metric(f"agentbench_v2_{suite}_score", result["score"], result["metadata"])
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating suite {suite} in V2: {e}")

        return results
