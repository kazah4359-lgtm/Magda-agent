import logging

from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

class OnlineRLUserBehaviorV4:
    """
    OpenClaw-RL Online RL from User Behavior v4.
    Extends the online reinforcement learning to include implicit signals from
    user interaction frequency and session duration.
    """

    def __init__(self, habit_tracker: HabitTracker, mirror_neurons: MirrorNeurons) -> None:
        """
        Initializes the learner with its dependencies.

        Args:
            habit_tracker: The system for tracking habit/skill usage.
            mirror_neurons: The sentiment and empathy processor for implicit feedback.
        """
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons

    async def process_session_metrics(
        self,
        frequency: int,
        duration_seconds: float,
        skill_name: str,
        user_id: int
    ) -> None:
        """
        Analyzes session metrics as implicit signals to adjust habit weights.
        High frequency and long duration indicate strong user engagement.

        Args:
            frequency (int): Number of interactions in the session.
            duration_seconds (float): Total session duration in seconds.
            skill_name (str): The skill being evaluated.
            user_id (int): The ID of the current user.
        """
        if frequency <= 0 or duration_seconds <= 0:
            return

        # Calculate an engagement score based on frequency and duration.
        # A simple heuristic: interactions per minute, scaled.
        duration_minutes = duration_seconds / 60.0
        interaction_rate = frequency / max(0.1, duration_minutes)

        # Base reward mapping from interaction rate.
        # High interaction rate -> positive reinforcement.
        base_reward = min(10.0, interaction_rate * 2.0)

        # Long sessions add a bonus
        if duration_minutes > 10.0:
            base_reward += 2.0

        reward = max(0.0, min(10.0, base_reward))

        if reward > 5.0:
            # Positive signal, reinforce habits
            self.habit_tracker.record_usage(
                input_text=f"Session metrics: {frequency} interactions over {duration_seconds} seconds",
                skill_used=skill_name,
                evaluation_score=reward,
                user_id=user_id
            )
            logging.info(f"OnlineRLUserBehaviorV4: Positive session signal (reward={reward:.2f}). Reinforced skill: {skill_name}")
        else:
            # Negative or low neutral signal, do not reinforce
            logging.info(f"OnlineRLUserBehaviorV4: Low session signal (reward={reward:.2f}). No usage recorded for {skill_name}.")
