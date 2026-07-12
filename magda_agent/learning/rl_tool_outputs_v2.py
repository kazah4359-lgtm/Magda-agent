import logging
from typing import Optional, List, Dict, Any

from magda_agent.learning.habits import HabitTracker
from magda_agent.user_model.model import UserModel

class OpenClawRLToolOutputsV2:
    """
    OpenClaw-RL Online Reinforcement Learning from Tool Outputs v2.
    Implements online reinforcement learning by tracking direct feedback
    signals from tool outputs, dynamically adjusting skill weights without
    user feedback.
    """

    def __init__(
        self,
        habit_tracker: HabitTracker,
        user_model: UserModel,
    ) -> None:
        """
        Initializes the learner with its dependencies.

        Args:
            habit_tracker: The system for tracking habit/skill usage.
            user_model: The user model to maintain user-specific preferences and skill weights.
        """
        self.habit_tracker = habit_tracker
        self.user_model = user_model

    async def process_tool_output_signal(
        self,
        tool_name: str,
        tool_output: str,
        action_context: str,
        user_id: int,
    ) -> None:
        """
        Analyzes the tool output to determine success or failure, and updates Q-values/weights dynamically.
        This provides implicit reinforcement learning without requiring direct user feedback.

        Args:
            tool_name (str): The name of the tool/skill that was executed.
            tool_output (str): The raw string output from the executed tool.
            action_context (str): A description of the action taken (used for habit tracking).
            user_id (int): The ID of the current user.
        """
        if not tool_output or not action_context:
            return

        # Simple heuristic for tool output success/failure
        # If output contains error keywords, consider it a failure. Otherwise success.
        output_lower = tool_output.lower()
        error_keywords = ["error", "exception", "failed", "traceback", "not found", "unauthorized", "invalid"]
        is_failure = any(kw in output_lower for kw in error_keywords)

        reward = 2.0 if is_failure else 8.0

        # Retrieve user model
        model_data = self.user_model.get_model(user_id)

        # Ensure skill_weights dictionary exists
        if "skill_weights" not in model_data:
            model_data["skill_weights"] = {}

        current_weight = model_data["skill_weights"].get(tool_name, 1.0)

        if reward > 5.0:
            # Positive signal (tool succeeded)
            # Increase weight by 0.2, max 2.0
            new_weight = min(2.0, current_weight + 0.2)
            model_data["skill_weights"][tool_name] = new_weight

            # Record usage in habit tracker
            self.habit_tracker.record_usage(
                input_text=action_context,
                skill_used=tool_name,
                evaluation_score=reward,
                user_id=user_id
            )
            logging.info(f"OpenClawRLToolOutputsV2: Positive tool output signal (reward={reward:.2f}). Reinforced skill: {tool_name} (weight: {current_weight:.2f} -> {new_weight:.2f})")
        else:
            # Negative signal (tool failed)
            # Decrease weight by 0.2, min 0.1
            new_weight = max(0.1, current_weight - 0.2)
            model_data["skill_weights"][tool_name] = new_weight
            logging.info(f"OpenClawRLToolOutputsV2: Negative tool output signal (reward={reward:.2f}). Penalized skill: {tool_name} (weight: {current_weight:.2f} -> {new_weight:.2f})")

        # Save preferences state dynamically
        if "rl_tool_preferences" not in model_data:
            model_data["rl_tool_preferences"] = {}
        model_data["rl_tool_preferences"]["last_tool_reward"] = reward

        # Save the model
        self.user_model.save_model(user_id, model_data)
        logging.info(f"OpenClawRLToolOutputsV2: Updated user model for user {user_id} with tool output feedback.")
