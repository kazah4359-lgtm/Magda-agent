import json
import logging
from typing import Optional
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.user_model.model import UserModel

class OpenClawInteractiveLearner:
    """
    OpenClaw-RL pattern interactive learner.
    Implements online reinforcement learning from next-state signals (user replies).
    """
    def __init__(
        self,
        habit_tracker: HabitTracker,
        mirror_neurons: MirrorNeurons,
        user_model: UserModel
    ) -> None:
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons
        self.user_model = user_model

    async def process_next_state_signal(self, user_reply: str, action_context: str, user_id: int, tool_output: Optional[str] = None) -> None:
        """
        Analyzes the user's reply as a next-state signal, and reinforces habits and updates preferences.

        Args:
            user_reply (str): The text of the user's reply.
            action_context (str): The context of the action that was taken.
            user_id (int): The user's ID.
            tool_output (Optional[str], optional): The output of the tool, if any. Defaults to None.
        """
        if not user_reply or not action_context:
            return

        signal_text = user_reply
        if tool_output:
            signal_text += f" [Tool Output: {tool_output}]"

        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(signal_text)
        model_data = self.user_model.get_model(user_id)

        if p_shift > 0.0:
            # Positive signal, reinforce the habit explicitly
            self.habit_tracker.record_usage(input_text=action_context, skill_used="rl_skill", evaluation_score=10.0, user_id=user_id)
            logging.info(f"OpenClaw-RL: Positive signal received (p_shift={p_shift:.2f}). Reinforced habit.")

            # Adjust communication style towards friendly if not present
            if "(friendly)" not in model_data.get("communication_style", ""):
                model_data["communication_style"] = f"{model_data.get('communication_style', 'default')} (friendly)"

        elif p_shift < 0.0:
            logging.info(f"OpenClaw-RL: Negative signal received (p_shift={p_shift:.2f}).")

            # Adjust communication style towards cautious if not present
            if "(cautious)" not in model_data.get("communication_style", ""):
                model_data["communication_style"] = f"{model_data.get('communication_style', 'default')} (cautious)"

        # Update preferences weight dynamically
        if "preferences" not in model_data:
            model_data["preferences"] = {}
        model_data["preferences"]["last_p_shift"] = p_shift

        # Save the updated user model back to disk
        self.user_model.save_model(user_id, model_data)
        logging.info(f"OpenClaw-RL: Updated user model for user {user_id}")
