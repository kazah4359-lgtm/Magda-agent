import logging
from typing import Any

from magda_agent.learning.openclaw_rl_rollout import OpenClawRLTrajectoryRolloutV6
from magda_agent.memory.context_engine import ContextPlugin


class OpenClawBehaviorPlugin(ContextPlugin):
    """
    Context Engine lifecycle plugin that logs multi-turn trajectories to update
    User behavior weights via OpenClawRLTrajectoryRolloutV6.
    """

    def __init__(self, rollout: OpenClawRLTrajectoryRolloutV6) -> None:
        """
        Initialize the plugin with the rollout tracker.

        Args:
            rollout: The RL rollout buffer and updater.
        """
        self.rollout = rollout

    def before_write(self, context: Any, user_id: int) -> Any:
        """
        Logs multi-turn trajectories to the rollout buffer.

        Args:
            context: The memory context being written.
            user_id: The user ID.

        Returns:
            The unmodified context.
        """
        skill_id = "unknown_skill"
        content = ""

        if isinstance(context, dict):
            skill_id = context.get("skill_id", skill_id)
            content = str(context.get("content", ""))
        elif hasattr(context, "skill_id") or hasattr(context, "content"):
            skill_id = getattr(context, "skill_id", skill_id)
            content = str(getattr(context, "content", ""))
        else:
            content = str(context)

        self.rollout.record_step(str(user_id), skill_id, content)
        return context

    def on_context_update(self, new_context: Any, user_id: int) -> None:
        """
        Checks for a reward signal and updates behavior weights.

        Args:
            new_context: The new context update containing potential rewards.
            user_id: The user ID.
        """
        reward = None

        if isinstance(new_context, dict):
            reward = new_context.get("reward")
        elif hasattr(new_context, "reward"):
            reward = getattr(new_context, "reward")

        if reward is not None:
            try:
                reward_val = float(reward)
                self.rollout.process_delayed_reward(str(user_id), reward_val)
            except (ValueError, TypeError):
                logging.warning(f"Invalid reward value: {reward}")
