import logging
from typing import Optional
from magda_agent.learning.habits import HabitTracker
from magda_agent.memory.storage import MemorySystem
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.emotions.engine import PADState

class OnlineLearner:
    """
    Online learning from user replies (next-state signal).
    Treats each user reply as a next-state signal. Positive signals reinforce approach,
    negative signals trigger reflection and adjustment.
    """
    def __init__(
        self,
        habit_tracker: HabitTracker,
        memory: MemorySystem,
        mirror_neurons: MirrorNeurons
    ) -> None:
        self.habit_tracker = habit_tracker
        self.memory = memory
        self.mirror_neurons = mirror_neurons

    async def process_feedback(self, user_reply: str, last_action_context: str, user_id: Optional[int] = None) -> None:
        """
        Analyzes the user's reply to an action, and reinforces or penalizes the habit.

        Args:
            user_reply (str): The text of the user's reply.
            last_action_context (str): The context of the action that was taken.
            user_id (Optional[int]): The user's ID.
        """
        if not user_reply or not last_action_context:
            return

        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_reply)

        if p_shift > 0.0:
            # Positive signal, reinforce the habit
            # Note: "online_learned_skill" is a placeholder for the actual skill used
            # For this simple heuristic, we just assume the last action was successful.
            self.habit_tracker.record_usage(last_action_context, "online_learned_skill", 10.0, user_id=user_id)
            logging.info(f"Online learning: Positive signal received (p_shift={p_shift:.2f}). Reinforced habit.")
        elif p_shift < 0.0:
            # Negative signal, trigger reflection
            content = f"Negative feedback received for action context '{last_action_context}'. User said: '{user_reply}'"
            await self.memory.add_memory(
                content=content,
                importance=0.8,
                emotional_state=PADState(p_shift, a_shift, d_shift),
                tags=["reflection", "online_learning", "negative_feedback"],
                user_id=user_id
            )
            logging.info(f"Online learning: Negative signal received (p_shift={p_shift:.2f}). Stored reflection.")
