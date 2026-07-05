import json
import logging
from typing import Optional, List
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.user_model.model import UserModel
from magda_agent.learning.lessons import TaskRecoveryLessons

class OpenClawInteractiveLearner:
    """
    OpenClaw-RL pattern interactive learner.
    Implements online reinforcement learning from next-state signals (user replies).
    """
    def __init__(
        self,
        habit_tracker: HabitTracker,
        mirror_neurons: MirrorNeurons,
        user_model: UserModel,
        recovery_lessons: Optional[TaskRecoveryLessons] = None
    ) -> None:
        """
        Initializes the OpenClawInteractiveLearner.

        Args:
            habit_tracker (HabitTracker): The tracker for agent habits.
            mirror_neurons (MirrorNeurons): The mirror neurons module for empathizing.
            user_model (UserModel): The persistent user model.
            recovery_lessons (Optional[TaskRecoveryLessons], optional): The recovery lessons generator. Defaults to None.
        """
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons
        self.user_model = user_model
        self.recovery_lessons = recovery_lessons

    async def process_next_state_signal(
        self,
        user_reply: str,
        action_context: str,
        user_id: Optional[str],
        tool_output: Optional[str] = None,
        skills_used: Optional[List[str]] = None
    ) -> None:
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

        signal_text: str = user_reply
        if tool_output:
            signal_text += f" [Tool Output: {tool_output}]"

        p_shift: float
        a_shift: float
        d_shift: float
        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(signal_text)
        model_data: dict = self.user_model.get_model(user_id)

        if p_shift > 0.0:
            # Positive signal, reinforce the habits explicitly
            skills: List[str] = skills_used or ["rl_skill"]

            # Update skill weights
            skill_weights = model_data.setdefault("skill_weights", {})
            for skill in skills:
                current_w = skill_weights.get(skill, 1.0)
                skill_weights[skill] = min(2.0, current_w + p_shift * 0.2)
                self.habit_tracker.record_usage(input_text=action_context, skill_used=skill, evaluation_score=10.0, user_id=user_id)

            logging.info(f"OpenClaw-RL: Positive signal received (p_shift={p_shift:.2f}). Reinforced habits: {skills}")

            # Adjust communication style towards friendly if not present
            if "(friendly)" not in model_data.get("communication_style", ""):
                model_data["communication_style"] = f"{model_data.get('communication_style', 'default')} (friendly)"

        elif p_shift < 0.0:
            logging.info(f"OpenClaw-RL: Negative signal received (p_shift={p_shift:.2f}).")

            # Update skill weights (penalty)
            if skills_used:
                skill_weights = model_data.setdefault("skill_weights", {})
                for skill in skills_used:
                    current_w = skill_weights.get(skill, 1.0)
                    skill_weights[skill] = max(0.1, current_w + p_shift * 0.2)

            # Directive signal: generate recovery lesson for significant negative feedback
            if p_shift <= -0.2 and self.recovery_lessons:
                logging.info("OpenClaw-RL: Significant negative signal. Generating recovery lesson.")
                await self.recovery_lessons.generate_and_store_lesson(
                    task_description=action_context,
                    failure_reason=user_reply,
                    user_id=user_id
                )

            # Adjust communication style towards cautious if not present
            if "(cautious)" not in model_data.get("communication_style", ""):
                model_data["communication_style"] = f"{model_data.get('communication_style', 'default')} (cautious)"

        # Update preferences weight dynamically
        if "preferences" not in model_data:
            model_data["preferences"] = {}
        model_data["preferences"]["last_p_shift"] = p_shift

        # Dynamic Behavior Adjustment based on implicit emotional shifts
        behavior_weights = model_data.setdefault("behavior_weights", {
            "exploration": 1.0,
            "verbosity": 1.0,
            "directness": 1.0
        })

        # Adjust exploration based on pleasure shift
        if p_shift > 0.0:
            behavior_weights["exploration"] = min(2.0, behavior_weights["exploration"] + p_shift * 0.5)
        elif p_shift < 0.0:
            behavior_weights["exploration"] = max(0.5, behavior_weights["exploration"] + p_shift * 0.5)

        # Adjust verbosity based on arousal shift
        if a_shift > 0.0:
            behavior_weights["verbosity"] = min(2.0, behavior_weights["verbosity"] + a_shift)
        elif a_shift < 0.0:
            behavior_weights["verbosity"] = max(0.5, behavior_weights["verbosity"] + a_shift)

        # Adjust directness based on dominance shift
        if d_shift > 0.0:
            behavior_weights["directness"] = min(2.0, behavior_weights["directness"] + d_shift)
        elif d_shift < 0.0:
            behavior_weights["directness"] = max(0.5, behavior_weights["directness"] + d_shift)

        model_data["behavior_weights"] = behavior_weights

        # Save the updated user model back to disk
        self.user_model.save_model(user_id, model_data)
        logging.info(f"OpenClaw-RL: Updated user model for user {user_id}")
