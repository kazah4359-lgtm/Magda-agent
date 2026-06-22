import logging
from typing import Optional, List

from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.user_model.model import UserModel

class OpenClawRLOnlineLearnerV2:
    """
    OpenClaw-RL Online Reinforcement Learning v2.
    Implements online learning from next-state signals, including user replies and tool outputs.
    """

    def __init__(
        self,
        habit_tracker: HabitTracker,
        mirror_neurons: MirrorNeurons,
        user_model: UserModel,
    ) -> None:
        """
        Initializes the learner with its dependencies.

        Args:
            habit_tracker: The system for tracking habit/skill usage.
            mirror_neurons: The sentiment and empathy processor for implicit feedback.
            user_model: The user model to maintain user-specific preferences.
        """
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons
        self.user_model = user_model

    async def process_next_state_signal(
        self,
        user_reply: str,
        action_context: str,
        user_id: int,
        tool_output: Optional[str] = None,
        skills_used: Optional[List[str]] = None,
    ) -> None:
        """
        Analyzes the subsequent state (user reply and tool output) and updates Q-values/weights dynamically.

        Args:
            user_reply (str): The user's reply string.
            action_context (str): A description of the action taken.
            user_id (int): The ID of the current user.
            tool_output (Optional[str], optional): The output from the executed tool. Defaults to None.
            skills_used (Optional[List[str]], optional): A list of skill names used. Defaults to None.
        """
        if not user_reply or not action_context:
            return

        signal_text = user_reply
        if tool_output:
            signal_text += f" [Tool Output: {tool_output}]"

        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(signal_text)

        skills = skills_used or ["rl_skill_v2"]

        # Dynamic Q-Value calculation inspired by RL
        # p_shift ranges between -1.0 to 1.0 (approximate).
        # We transform this into a score between 0.0 and 10.0
        base_reward = (p_shift + 1.0) * 5.0

        # Give bonus if there's a valid tool output and positive empathy shift
        if tool_output and p_shift > 0.0:
            base_reward += 2.0

        reward = max(0.0, min(10.0, base_reward))

        # Retrieve user model
        model_data = self.user_model.get_model(user_id)

        if reward > 5.0:
            # Positive signal, reinforce habits
            for skill in skills:
                self.habit_tracker.record_usage(
                    input_text=action_context,
                    skill_used=skill,
                    evaluation_score=reward,
                    user_id=user_id
                )
            logging.info(f"OpenClawRLV2: Positive signal received (reward={reward:.2f}). Reinforced skills: {skills}")

            if "(confident)" not in model_data.get("communication_style", ""):
                model_data["communication_style"] = f"{model_data.get('communication_style', 'default')} (confident)"
        else:
            # Negative or low neutral signal
            logging.info(f"OpenClawRLV2: Low/Negative signal (reward={reward:.2f}). No usage recorded.")

            if "(attentive)" not in model_data.get("communication_style", ""):
                model_data["communication_style"] = f"{model_data.get('communication_style', 'default')} (attentive)"

        # Save preferences state dynamically
        if "rl_v2_preferences" not in model_data:
            model_data["rl_v2_preferences"] = {}
        model_data["rl_v2_preferences"]["last_reward"] = reward

        # Save the model
        self.user_model.save_model(user_id, model_data)
        logging.info(f"OpenClawRLV2: Updated user model for user {user_id} with next-state feedback.")
