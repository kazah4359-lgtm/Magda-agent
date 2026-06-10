import logging
from typing import Optional, Dict
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
        self.skill_weights: Dict[str, float] = {}

    async def process_feedback(self, user_reply: str, action_context: str, user_id: Optional[int] = None, explicit_score: Optional[float] = None, tool_success: bool = False, skill_used: str = "rl_feedback_skill") -> None:
        """
        Processes explicit and implicit feedback signals to adjust habit weights.

        Args:
            user_reply (str): The user's reply.
            action_context (str): The context of the action taken.
            user_id (Optional[int]): The ID of the user.
            explicit_score (Optional[float]): Explicit user feedback score (e.g. from 0 to 10).
            tool_success (bool): Whether the tool execution was successful.
            skill_used (str): The name of the skill used.
        """
        if not user_reply or not action_context:
            return

        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_reply)

        if explicit_score is not None:
            base_score = explicit_score
        else:
            base_score = (p_shift + 1.0) * 5.0

        weight = base_score
        if tool_success:
            weight += 2.0

        if skill_used not in self.skill_weights:
            self.skill_weights[skill_used] = 1.0

        if weight >= 8.0:
            self.skill_weights[skill_used] += 0.1
            self.habit_tracker.record_usage(
                input_text=action_context,
                skill_used=skill_used,
                evaluation_score=weight,
                user_id=user_id
            )
            logging.info(f"Online RL: Positive feedback (weight={weight:.2f}). Recorded usage. New weight for {skill_used}: {self.skill_weights[skill_used]:.2f}")
        else:
            self.skill_weights[skill_used] = max(0.1, self.skill_weights[skill_used] - 0.1)
            logging.info(f"Online RL: Negative/Neutral feedback (weight={weight:.2f}). No usage recorded. New weight for {skill_used}: {self.skill_weights[skill_used]:.2f}")
