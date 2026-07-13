import logging
from typing import Optional, Dict, Any
from magda_agent.learning.rl_metrics_v4 import RLMetricsTrackerV4

class RLNextStateSignalsV5:
    """
    OpenClaw RL Next-State Signals v5.
    Handles next-state signals dynamically and updates Q-values using RLMetricsTracker.
    Inspired by OpenClaw-RL (June 2026).
    """

    def __init__(
        self,
        metrics_tracker: RLMetricsTrackerV4,
        alpha: float = 0.1,
        gamma: float = 0.9
    ) -> None:
        """
        Initializes the RL Next-State Signals module.

        Args:
            metrics_tracker (RLMetricsTrackerV4): Tracker for logging rewards and Q-value updates.
            alpha (float): Learning rate (0.0 to 1.0).
            gamma (float): Discount factor for next-state Q-values (0.0 to 1.0).
        """
        self.metrics_tracker = metrics_tracker
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: Dict[str, float] = {}
        logging.info("Initialized RLNextStateSignalsV5")

    def parse_reward(self, text: str) -> float:
        """
        Parses a text signal to extract a reward score.

        Args:
            text (str): The text signal (e.g., user reply or tool output summary).

        Returns:
            float: A reward score between -1.0 and 1.0.
        """
        text_lower = text.lower()
        # Basic normalization
        words = text_lower.replace("'", "").replace(".", "").replace(",", "").replace("!", "").split()

        positive_words = ["good", "great", "thanks", "awesome", "yes", "correct", "success", "resolved"]
        negative_words = ["bad", "terrible", "wrong", "no", "incorrect", "fail", "error", "failed"]

        if any(w in positive_words for w in words):
            return 1.0
        elif any(w in negative_words for w in words):
            return -1.0
        return 0.0

    def process_signal(
        self,
        skill_id: str,
        reward: float,
        next_state_max_q: float = 0.0,
        user_id: Optional[str] = None
    ) -> float:
        """
        Processes a numerical reward signal and updates the Q-value for a skill.

        Args:
            skill_id (str): The identifier of the skill used.
            reward (float): The immediate reward received.
            next_state_max_q (float): The estimated maximum Q-value of the resulting next state.
            user_id (Optional[str]): The optional ID of the user.

        Returns:
            float: The updated Q-value for the skill.
        """
        current_q = self.q_table.get(skill_id, 0.0)

        # Q-learning temporal difference update rule:
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        td_target = reward + self.gamma * next_state_max_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error

        self.q_table[skill_id] = new_q

        # Log metrics to the longitudinal tracker
        self.metrics_tracker.log_reward(reward, skill_id, user_id=user_id)
        self.metrics_tracker.log_weight_delta(new_q - current_q, skill_id, user_id=user_id)

        logging.info(
            f"RLNextStateSignalsV5: Updated Q-value for '{skill_id}' to {new_q:.4f} "
            f"(Reward: {reward}, Next Q: {next_state_max_q}, TD Error: {td_error:.4f})"
        )
        return new_q

    def handle_interaction(
        self,
        skill_id: str,
        signal_text: str,
        next_state_max_q: float = 0.0,
        user_id: Optional[str] = None
    ) -> float:
        """
        High-level method to handle an interaction by parsing a text signal and updating Q-values.

        Args:
            skill_id (str): The identifier of the skill.
            signal_text (str): The text of the signal (e.g., user reply).
            next_state_max_q (float): The estimated maximum Q-value of the next state.
            user_id (Optional[str]): The optional ID of the user.

        Returns:
            float: The updated Q-value.
        """
        reward = self.parse_reward(signal_text)
        return self.process_signal(skill_id, reward, next_state_max_q, user_id=user_id)

    def get_q_value(self, skill_id: str) -> float:
        """
        Retrieves the current Q-value for a given skill.

        Args:
            skill_id (str): The identifier of the skill.

        Returns:
            float: The current Q-value.
        """
        return self.q_table.get(skill_id, 0.0)
