import logging
from typing import Dict, Optional


class InteractiveLearner:
    """
    OpenClaw-RL pattern interactive learner.
    Implements online reinforcement learning from next-state signals (user replies).
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Initializes the InteractiveLearner.

        Args:
            initial_weights (Optional[Dict[str, float]]): Initial weights for skills.
        """
        self.learning_state: Dict[str, float] = dict(initial_weights) if initial_weights else {}
        logging.info("Initialized InteractiveLearner")

    def analyze_signal(self, reply_text: str) -> float:
        """
        Analyzes the user's reply as a next-state signal.
        Uses basic heuristics to determine reward.

        Args:
            reply_text (str): The text of the user's reply.

        Returns:
            float: A reward score, positive for positive sentiment, negative for negative sentiment.
        """
        reply_lower = reply_text.lower()
        if "good" in reply_lower or "great" in reply_lower or "yes" in reply_lower or "thanks" in reply_lower:
            return 1.0
        elif "bad" in reply_lower or "wrong" in reply_lower or "no" in reply_lower or "terrible" in reply_lower:
            return -1.0
        return 0.0

    async def process_interaction(
        self,
        user_reply: str,
        skill_name: str = "default_skill",
        user_id: Optional[int] = None
    ) -> None:
        """
        Analyzes the user's reply and updates learning state.

        Args:
            user_reply (str): The text of the user's reply.
            skill_name (str): The name of the skill to adjust based on the reply.
            user_id (Optional[int]): The ID of the user.
        """
        if not user_reply:
            return

        reward = self.analyze_signal(user_reply)

        if skill_name not in self.learning_state:
            self.learning_state[skill_name] = 1.0

        adjustment_rate = 0.2
        self.learning_state[skill_name] += reward * adjustment_rate

        # Clamp between 0.1 and 10.0
        self.learning_state[skill_name] = max(0.1, min(10.0, self.learning_state[skill_name]))

        logging.info(f"Updated interactive learning state for skill '{skill_name}' to {self.learning_state[skill_name]:.2f} based on reward {reward}")

    def get_state(self) -> Dict[str, float]:
        """
        Retrieves the current learning state (skill weights).

        Returns:
            Dict[str, float]: The current state of learned skill weights.
        """
        return self.learning_state
