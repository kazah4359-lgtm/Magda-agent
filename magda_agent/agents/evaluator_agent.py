import logging
from typing import Optional, Dict, Any

from magda_agent.metacognition.evaluator import Evaluator
from magda_agent.metacognition.confidence import ConfidenceCalibrator
from magda_agent.learning.habits import HabitTracker
from magda_agent.planning.planner import Planner

class EvaluatorAgent:
    """
    Agent responsible for evaluating the generated response.
    """
    def __init__(
        self,
        evaluator: Optional[Evaluator] = None,
        confidence_calibrator: Optional[ConfidenceCalibrator] = None,
        habit_tracker: Optional[HabitTracker] = None,
        planner: Optional[Planner] = None
    ):
        self.evaluator = evaluator
        self.confidence_calibrator = confidence_calibrator
        self.habit_tracker = habit_tracker
        self.planner = planner

    async def evaluate(self, user_input: str, response: str, user_id: Optional[str] = None):
        """
        Evaluates the interaction and records habits.
        """
        if not self.evaluator:
            return

        await self.evaluator.evaluate_response(user_input, response)

        if self.confidence_calibrator and self.evaluator.last_evaluation and self.confidence_calibrator.last_confidence is not None:
            actual_score = self.evaluator.last_evaluation.get("average_score", 0.0)
            self.confidence_calibrator.track_calibration(self.confidence_calibrator.last_confidence, actual_score)

        if self.habit_tracker and self.evaluator.last_evaluation:
            avg_score = self.evaluator.last_evaluation.get("average_score", 0.0)
            if self.planner and self.planner.completed_steps:
                for step in self.planner.completed_steps:
                    skill = step.get("skill")
                    if skill:
                        self.habit_tracker.record_usage(user_input, skill, float(avg_score), user_id=user_id)
