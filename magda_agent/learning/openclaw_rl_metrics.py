import logging
from typing import Dict, List, Any, Optional

class OpenClawRLMetrics:
    """
    OpenClaw RL Metrics for visualization.
    Tracks Q-value updates and rewards to provide a visual representation
    of the reinforcement learning system's progress.
    Inspired by OpenClaw Canvas Live Visualization trends.
    """

    def __init__(self) -> None:
        """
        Initializes the OpenClawRLMetrics tracker.
        """
        self.q_values: Dict[str, float] = {}
        self.recent_rewards: List[Dict[str, Any]] = []
        self.max_recent_rewards = 20
        logging.info("Initialized OpenClawRLMetrics")

    def update_q_value(self, skill_id: str, new_q: float) -> None:
        """
        Updates the tracked Q-value for a specific skill.

        Args:
            skill_id (str): The identifier of the skill.
            new_q (float): The updated Q-value.
        """
        self.q_values[skill_id] = new_q
        logging.debug(f"OpenClawRLMetrics: Updated Q-value for {skill_id} to {new_q:.4f}")

    def add_reward(self, skill_id: str, reward: float, user_id: Optional[str] = None) -> None:
        """
        Adds a reward signal to the recent history.

        Args:
            skill_id (str): The identifier of the skill.
            reward (float): The reward value.
            user_id (Optional[str]): The ID of the user providing feedback.
        """
        entry = {
            "skill_id": skill_id,
            "reward": reward,
            "user_id": user_id
        }
        self.recent_rewards.append(entry)
        if len(self.recent_rewards) > self.max_recent_rewards:
            self.recent_rewards.pop(0)
        logging.debug(f"OpenClawRLMetrics: Added reward {reward} for {skill_id}")

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Formats the current metrics into a dictionary suitable for canvas visualization.

        Returns:
            Dict[str, Any]: The visualization data including Q-values, recent rewards,
                          and summary stats.
        """
        average_q = sum(self.q_values.values()) / len(self.q_values) if self.q_values else 0.0
        return {
            "q_values": self.q_values,
            "recent_rewards": self.recent_rewards,
            "average_q": average_q,
            "reward_count": len(self.recent_rewards),
            "status": "active"
        }
