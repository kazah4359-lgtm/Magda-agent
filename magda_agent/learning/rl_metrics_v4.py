import logging
from typing import Optional, List, Dict, Any
from magda_agent.metacognition.tracker import QualityTracker

class RLMetricsTrackerV4:
    """
    OpenClaw RL Metrics Tracker v4.
    Tracks longitudinal reinforcement learning metrics such as reward trends
    and skill weight stability over time.
    """

    def __init__(self, quality_tracker: QualityTracker) -> None:
        """
        Initializes the RL Metrics Tracker.

        Args:
            quality_tracker (QualityTracker): The persistent tracker for logging metrics.
        """
        self.quality_tracker = quality_tracker
        logging.info("Initialized RLMetricsTrackerV4")

    def log_reward(self, reward: float, skill_name: str, user_id: Optional[str] = None) -> None:
        """
        Logs a reward signal for a specific skill.

        Args:
            reward (float): The reward value received.
            skill_name (str): The name of the skill the reward is for.
            user_id (Optional[str]): The ID of the user providing the feedback.
        """
        metadata = {
            "skill_name": skill_name,
            "user_id": user_id,
            "type": "reward"
        }
        self.quality_tracker.log_metric("rl_reward_v4", reward, metadata)
        logging.debug(f"RLMetricsTrackerV4: Logged reward {reward} for skill {skill_name}")

    def log_weight_delta(self, delta: float, skill_name: str, user_id: Optional[str] = None) -> None:
        """
        Logs a change in a skill's weight.

        Args:
            delta (float): The change in the skill's weight.
            skill_name (str): The name of the skill.
            user_id (Optional[str]): The ID of the user providing the feedback.
        """
        metadata = {
            "skill_name": skill_name,
            "user_id": user_id,
            "type": "weight_delta"
        }
        self.quality_tracker.log_metric("rl_weight_delta_v4", delta, metadata)
        logging.debug(f"RLMetricsTrackerV4: Logged weight delta {delta} for skill {skill_name}")

    def get_reward_trend(self, skill_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves the reward trend for a specific skill.

        Args:
            skill_name (str): The name of the skill.
            limit (int): The number of recent entries to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of recent reward entries for the skill.
        """
        all_rewards = self.quality_tracker.get_metrics("rl_reward_v4", limit=limit * 5)
        # Filter by skill_name manually as QualityTracker's get_metrics doesn't filter metadata
        filtered = [r for r in all_rewards if r.get("skill_name") == skill_name][:limit]
        return filtered

    def calculate_average_reward(self, skill_name: str, limit: int = 10) -> Optional[float]:
        """
        Calculates the average reward for a skill over recent interactions.

        Args:
            skill_name (str): The name of the skill.
            limit (int): The number of recent entries to consider.

        Returns:
            Optional[float]: The average reward, or None if no entries exist.
        """
        recent_rewards = self.get_reward_trend(skill_name, limit=limit)
        if not recent_rewards:
            return None

        total = sum(float(r.get("value", 0.0)) for r in recent_rewards)
        return total / len(recent_rewards)
