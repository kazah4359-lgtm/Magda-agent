import logging
from typing import Dict, Any

class OnlineRLFeedbackLoop:
    """
    Component that captures user feedback signals (e.g. positive/negative replies)
    and maps them to Q-value updates for skills.
    Inspired by OpenClaw-RL.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Initializes the OnlineRLFeedbackLoop.

        Args:
            db_path (str): The path to the SQLite database. Defaults to in-memory.
        """
        self.q_table: Dict[str, float] = {}
        logging.info("Initialized OnlineRLFeedbackLoop")

    def process_feedback(self, skill_id: str, user_reply: str) -> None:
        """
        Processes user feedback and updates the Q-value for a given skill.

        Args:
            skill_id (str): The identifier of the skill used.
            user_reply (str): The text of the user's reply.
        """
        reward = self._map_reply_to_reward(user_reply)

        current_q = self.q_table.get(skill_id, 0.0)

        # Simple Q-learning update rule with alpha=0.1, gamma=0 (immediate reward only)
        alpha = 0.1
        new_q = current_q + alpha * (reward - current_q)

        self.q_table[skill_id] = new_q
        logging.info(f"Updated Q-value for {skill_id}: {new_q:.2f} (Reward: {reward})")

    def _map_reply_to_reward(self, reply: str) -> float:
        """
        Maps a user reply text to a numerical reward score.

        Args:
            reply (str): The user reply.

        Returns:
            float: The reward score.
        """
        reply_lower = reply.lower()

        # Check negative first, but make sure it's not matching 'know' incorrectly due to 'no'
        # Tokenize by word
        words = reply_lower.replace("'", "").replace(".", "").replace(",", "").replace("!", "").split()

        if any(w in ["good", "great", "thanks", "awesome", "yes"] for w in words):
            return 1.0
        elif any(w in ["bad", "terrible", "wrong", "no"] for w in words):
            return -1.0
        else:
            return 0.0

    def get_q_value(self, skill_id: str) -> float:
        """
        Retrieves the current Q-value for a given skill.

        Args:
            skill_id (str): The identifier of the skill.

        Returns:
            float: The current Q-value.
        """
        return self.q_table.get(skill_id, 0.0)
