import logging
from typing import Optional, Dict
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

class OnlineRLFeedbackLoop:
    """
    Online Reinforcement Learning loop for adjusting agent behavior parameters
    based on immediate explicit and implicit user feedback without separate annotation.
    """
    def __init__(self, habit_tracker: HabitTracker, mirror_neurons: MirrorNeurons) -> None:
        """
        Initializes the OnlineRLFeedbackLoop.

        Args:
            habit_tracker (HabitTracker): Tracks and records skill usage.
            mirror_neurons (MirrorNeurons): Evaluates implicit feedback (emotional shifts).
        """
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons
        self.reward_parameters: Dict[str, float] = {}

    async def adjust_behavior(
        self,
        user_reply: str,
        action_context: str,
        user_id: Optional[int] = None,
        explicit_score: Optional[float] = None,
        skill_used: str = "rl_feedback_skill"
    ) -> None:
        """
        Adjusts reward parameters and behavior based on explicit and implicit feedback.

        Args:
            user_reply (str): The immediate user response.
            action_context (str): Context of the agent's action.
            user_id (Optional[int]): User ID.
            explicit_score (Optional[float]): Explicit numeric feedback from the user.
            skill_used (str): Skill used for the action.
        """
        if not user_reply or not action_context:
            return

        # Implicit feedback via mirror neurons
        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_reply)

        if explicit_score is not None:
            score = explicit_score
        else:
            score = (p_shift + 1.0) * 5.0

        if skill_used not in self.reward_parameters:
            self.reward_parameters[skill_used] = 1.0

        if score >= 7.0:
            self.reward_parameters[skill_used] += 0.2
            self.habit_tracker.record_usage(
                input_text=action_context,
                skill_used=skill_used,
                evaluation_score=score,
                user_id=user_id
            )
            logging.info(f"RL Feedback: Positive adjustment. New param for {skill_used}: {self.reward_parameters[skill_used]:.2f}")
        else:
            self.reward_parameters[skill_used] = max(0.1, self.reward_parameters[skill_used] - 0.2)
            logging.info(f"RL Feedback: Negative adjustment. New param for {skill_used}: {self.reward_parameters[skill_used]:.2f}")
