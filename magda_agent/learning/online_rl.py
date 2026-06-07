import logging
from typing import Optional
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

class OnlineRLIntegrator:
    """
    Online RL Feedback Integration.
    Collects feedback signals and adjusts habit weights dynamically.
    """
    def __init__(self, habit_tracker: HabitTracker, mirror_neurons: MirrorNeurons) -> None:
        """
        Initializes the OnlineRLIntegrator.

        Args:
            habit_tracker (HabitTracker): The habit tracker.
            mirror_neurons (MirrorNeurons): The mirror neurons for implicit feedback.
        """
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons

    async def process_feedback(self, user_reply: str, action_context: str, user_id: Optional[int] = None) -> None:
        """
        Processes explicit and implicit feedback signals to adjust habit weights.

        Args:
            user_reply (str): The user's reply.
            action_context (str): The context of the action taken.
            user_id (Optional[int]): The ID of the user.
        """
        if not user_reply or not action_context:
            return

        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_reply)

        # Calculate a dynamic weight based on the pleasure shift.
        # Scale the p_shift (-1.0 to 1.0) to a score (0 to 10).
        weight = (p_shift + 1.0) * 5.0

        if weight >= 8.0:
            self.habit_tracker.record_usage(
                input_text=action_context,
                skill_used="rl_feedback_skill",
                evaluation_score=weight,
                user_id=user_id
            )
            logging.info(f"Online RL: Positive feedback (weight={weight:.2f}). Recorded usage.")
        else:
            logging.info(f"Online RL: Negative/Neutral feedback (weight={weight:.2f}). No usage recorded.")
