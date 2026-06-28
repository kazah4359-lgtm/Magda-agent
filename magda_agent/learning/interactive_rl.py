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
        reply_lower: str = reply_text.lower()
        if any(word in reply_lower for word in ["good", "great", "yes", "thanks", "awesome", "excellent", "amazing"]):
            return 1.0
        elif any(word in reply_lower for word in ["bad", "wrong", "no", "terrible", "awful", "horrible", "stop", "halt", "cancel"]):
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

        reward: float = self.analyze_signal(user_reply)

        if skill_name not in self.learning_state:
            self.learning_state[skill_name] = 1.0

        adjustment_rate: float = 0.2
        self.learning_state[skill_name] += reward * adjustment_rate

        # Clamp between 0.1 and 10.0
        current_weight: float = self.learning_state[skill_name]
        self.learning_state[skill_name] = max(0.1, min(10.0, current_weight))

        logging.info(f"Updated interactive learning state for skill '{skill_name}' to {self.learning_state[skill_name]:.2f} based on reward {reward}")

    async def process_batch_interactions(
        self,
        interactions: list[tuple[str, str]],
        user_id: Optional[int] = None
    ) -> None:
        """
        Processes a batch of interactions concurrently or sequentially to update learning state.

        Args:
            interactions (list[tuple[str, str]]): A list of tuples containing (user_reply, skill_name).
            user_id (Optional[int]): The ID of the user.
        """
        for user_reply, skill_name in interactions:
            await self.process_interaction(user_reply, skill_name, user_id)

    def get_state(self) -> Dict[str, float]:
        """
        Retrieves the current learning state (skill weights).

        Returns:
            Dict[str, float]: The current state of learned skill weights.
        """
        return self.learning_state
