import logging
from typing import Dict, List, Any, Optional
from magda_agent.learning.openclaw_rl_metrics import OpenClawRLMetrics

logger = logging.getLogger(__name__)

class RLCanvasMetricsExporter:
    """
    RLCanvasMetricsExporter handles the transformation and formatting of
    reinforcement learning metrics for direct ingestion by live Canvas UIs.

    It converts tracked Q-values and raw rewards history into normalized metrics,
    calculates moving averages, detects trends, and aggregates behavioral metrics.
    """

    def __init__(self, metrics: OpenClawRLMetrics) -> None:
        """
        Initializes the exporter with a reference to the active OpenClawRLMetrics tracker.

        Args:
            metrics (OpenClawRLMetrics): The underlying RL metrics instance to export from.
        """
        self.metrics = metrics

    def get_normalized_q_values(self) -> Dict[str, float]:
        """
        Normalizes Q-values to a standard scale of [0.0, 1.0] across all tracked skills.
        If all Q-values are identical, they are mapped to 0.5.

        Returns:
            Dict[str, float]: Normalized Q-values mapped by skill identifier.
        """
        q_vals = self.metrics.q_values
        if not q_vals:
            return {}

        min_val = min(q_vals.values())
        max_val = max(q_vals.values())
        diff = max_val - min_val

        if diff == 0.0:
            return {skill: 0.5 for skill in q_vals}

        return {skill: (val - min_val) / diff for skill, val in q_vals.items()}

    def get_moving_average_reward(self, window_size: int = 5) -> float:
        """
        Calculates the moving average reward over the specified window size.

        Args:
            window_size (int): The number of recent rewards to consider. Defaults to 5.

        Returns:
            float: The moving average reward, or 0.0 if no rewards are present.
        """
        rewards = self.metrics.recent_rewards
        if not rewards:
            return 0.0

        window = rewards[-window_size:]
        total_reward = sum(entry["reward"] for entry in window if "reward" in entry)
        return total_reward / len(window)

    def detect_reward_trend(self, split_ratio: float = 0.5) -> str:
        """
        Detects whether the reward signal trend is "improving", "declining", or "stable"
        by comparing the average of the older half to the newer half of recent rewards.

        Args:
            split_ratio (float): The ratio at which to split the history. Defaults to 0.5.

        Returns:
            str: One of "improving", "declining", "stable", or "insufficient_data".
        """
        rewards = self.metrics.recent_rewards
        if len(rewards) < 4:
            return "insufficient_data"

        split_idx = int(len(rewards) * split_ratio)
        older_half = rewards[:split_idx]
        newer_half = rewards[split_idx:]

        avg_older = sum(entry["reward"] for entry in older_half) / len(older_half)
        avg_newer = sum(entry["reward"] for entry in newer_half) / len(newer_half)

        difference = avg_newer - avg_older
        if difference > 0.05:
            return "improving"
        elif difference < -0.05:
            return "declining"
        else:
            return "stable"

    def get_skill_activity_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Aggregates activity metrics per skill, tracking occurrence counts and
        average reward achieved per skill.

        Returns:
            Dict[str, Dict[str, Any]]: Aggregated skills stats including count and avg_reward.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for entry in self.metrics.recent_rewards:
            skill = entry.get("skill_id", "unknown")
            reward = entry.get("reward", 0.0)

            if skill not in stats:
                stats[skill] = {"count": 0, "total_reward": 0.0}

            stats[skill]["count"] += 1
            stats[skill]["total_reward"] += reward

        formatted_stats: Dict[str, Dict[str, Any]] = {}
        for skill, data in stats.items():
            count = data["count"]
            formatted_stats[skill] = {
                "count": count,
                "average_reward": data["total_reward"] / count if count > 0 else 0.0
            }

        return formatted_stats

    def export_canvas_payload(self) -> Dict[str, Any]:
        """
        Transforms and packages the metrics into a highly-structured payload
        ready for live Canvas UI ingestion.

        Returns:
            Dict[str, Any]: The structured UI visualization state.
        """
        try:
            raw_data = self.metrics.get_visualization_data()
            normalized_q = self.get_normalized_q_values()
            moving_avg_5 = self.get_moving_average_reward(window_size=5)
            trend = self.detect_reward_trend()
            skill_stats = self.get_skill_activity_metrics()

            return {
                "status": raw_data.get("status", "active"),
                "total_rewards_received": raw_data.get("reward_count", 0),
                "global_average_q": raw_data.get("average_q", 0.0),
                "moving_average_reward_5": moving_avg_5,
                "trajectory_trend": trend,
                "skills_coverage": {
                    skill: {
                        "q_value": raw_val,
                        "normalized_q_value": normalized_q.get(skill, 0.5),
                        "activity": skill_stats.get(skill, {}).get("count", 0),
                        "avg_reward": skill_stats.get(skill, {}).get("average_reward", 0.0)
                    }
                    for skill, raw_val in raw_data.get("q_values", {}).items()
                },
                "raw_rewards_trajectory": raw_data.get("recent_rewards", [])
            }
        except Exception as e:
            logger.error(f"Error exporting canvas metrics: {e}")
            return {
                "status": "error",
                "error_message": str(e)
            }
