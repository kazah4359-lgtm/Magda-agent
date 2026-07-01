import logging
import json
from typing import Optional, Dict, List
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

class OnlineFeedbackRL:
    """
    Updates tool execution weights based on user feedback.
    Inspired by OpenClaw trends: Online learning from interactions.
    """
    def __init__(self, habit_tracker: HabitTracker, mirror_neurons: MirrorNeurons) -> None:
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons
        self.skill_weights: Dict[str, float] = {}

    async def process_feedback(
        self,
        user_reply: str,
        action_context: str,
        skills_used: List[str],
        user_id: Optional[int] = None
    ) -> None:
        """
        Parses user feedback and adjusts weights for the skills used.
        """
        if not user_reply or not skills_used:
            return

        # Implicit feedback via sentiment analysis
        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_reply)

        # Heuristic: convert PAD shift to a scalar reward [0, 10]
        # P shift of 1.0 -> 10.0, P shift of -1.0 -> 0.0
        reward = (p_shift + 1.0) * 5.0

        logging.info(f"OnlineFeedbackRL: Received feedback '{user_reply[:30]}...', reward: {reward:.2f}")

        for skill in skills_used:
            if skill not in self.skill_weights:
                self.skill_weights[skill] = 1.0

            if reward >= 7.0:
                # Reinforce
                self.skill_weights[skill] += 0.1
                # Also record in habit tracker if strongly positive
                if reward >= 8.5:
                    self.habit_tracker.record_usage(
                        input_text=action_context,
                        skill_used=skill,
                        evaluation_score=reward,
                        user_id=user_id
                    )
            elif reward <= 3.0:
                # Penalize
                self.skill_weights[skill] = max(0.1, self.skill_weights[skill] - 0.2)

            logging.info(f"OnlineFeedbackRL: Updated weight for {skill}: {self.skill_weights[skill]:.2f}")

    def get_skill_weight(self, skill_name: str) -> float:
        return self.skill_weights.get(skill_name, 1.0)
