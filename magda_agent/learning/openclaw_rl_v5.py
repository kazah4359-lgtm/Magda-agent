import logging
from typing import Optional, Dict

class OnlineRLIntegrator:
    """
    OpenClaw-RL Online Reinforcement Learning Module v5.
    Updates skill_weights based on user feedback.
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None, initial_behavior: Optional[Dict[str, float]] = None) -> None:
        """
        Initializes the OnlineRLIntegrator.

        Args:
            initial_weights (Optional[Dict[str, float]]): Initial weights for skills.
            initial_behavior (Optional[Dict[str, float]]): Initial behavior parameters for the agent.
        """
        self.skill_weights: Dict[str, float] = initial_weights or {}
        self.behavior_parameters: Dict[str, float] = initial_behavior or {
            "exploration": 1.0,
            "verbosity": 1.0,
            "directness": 1.0
        }
        logging.info("Initialized OnlineRLIntegrator")

    def parse_feedback(self, feedback: str) -> float:
        """
        Parses user feedback to extract a sentiment or reward score.
        Uses simple heuristics for demonstration in lieu of a full language model.

        Args:
            feedback (str): The user's feedback text.

        Returns:
            float: A reward score, typically between -1.0 and 1.0.
        """
        feedback_lower = feedback.lower()
        if "good" in feedback_lower or "great" in feedback_lower or "thanks" in feedback_lower or "excellent" in feedback_lower:
            return 1.0
        elif "bad" in feedback_lower or "wrong" in feedback_lower or "terrible" in feedback_lower or "no" in feedback_lower:
            return -1.0
        return 0.0

    async def process_feedback(
        self,
        user_feedback: str,
        action_context: str,
        user_id: Optional[int] = None,
        skill_used: str = "default_skill"
    ) -> None:
        """
        Processes user feedback and adjusts the weight of the skill used.

        Args:
            user_feedback (str): The feedback provided by the user.
            action_context (str): The context in which the action was taken.
            user_id (Optional[int]): The ID of the user providing feedback.
            skill_used (str): The name of the skill that was used to generate the response.
        """
        if not user_feedback:
            return

        reward = self.parse_feedback(user_feedback)

        if skill_used not in self.skill_weights:
            self.skill_weights[skill_used] = 1.0

        # Adjust weight based on reward
        adjustment_factor = 0.1
        self.skill_weights[skill_used] += reward * adjustment_factor

        # Ensure weight doesn't go below a minimum threshold
        self.skill_weights[skill_used] = max(0.1, self.skill_weights[skill_used])

        logging.info(f"Updated weight for skill '{skill_used}' to {self.skill_weights[skill_used]:.2f} based on feedback reward {reward}")

        # Update behavior parameters dynamically
        behavior_adjustment = reward * 0.1
        for param in self.behavior_parameters:
            new_val = self.behavior_parameters[param] + behavior_adjustment
            self.behavior_parameters[param] = max(0.5, min(2.0, new_val))
            logging.info(f"Updated behavior parameter '{param}' to {self.behavior_parameters[param]:.2f}")
