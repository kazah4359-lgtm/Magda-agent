import logging
from typing import Optional, List
from magda_agent.learning.habits import HabitTracker
from magda_agent.emotions.mirror_neurons import MirrorNeurons

class RLSignalProcessor:
    """
    OpenClaw-RL pattern signal processor.
    Implements online reinforcement learning from next-state signals like user replies and tool outputs.
    """
    def __init__(
        self,
        habit_tracker: HabitTracker,
        mirror_neurons: MirrorNeurons
    ) -> None:
        self.habit_tracker = habit_tracker
        self.mirror_neurons = mirror_neurons

    async def process_signal(
        self,
        user_reply: str,
        action_context: str,
        user_id: Optional[int] = None,
        tool_output: Optional[str] = None,
        skills_used: Optional[List[str]] = None
    ) -> None:
        """
        Analyzes the user's reply and tool output as a next-state signal,
        and uses MirrorNeurons to evaluate it and update HabitTracker.

        Args:
            user_reply (str): The text of the user's reply.
            action_context (str): The context of the action that was taken.
            user_id (Optional[int]): The user's ID.
            tool_output (Optional[str], optional): The output of the tool, if any. Defaults to None.
            skills_used (Optional[List[str]], optional): List of skills used. Defaults to ["rl_skill"].
        """
        if not user_reply or not action_context:
            return

        signal_text = user_reply
        if tool_output:
            signal_text += f" [Tool Output: {tool_output}]"

        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(signal_text)

        if p_shift > 0.0:
            skills = skills_used or ["rl_skill"]
            # Convert p_shift (0 to 1) to a score out of 10.
            score = (p_shift + 1.0) * 5.0
            if tool_output:
                score += 1.0
            score = min(10.0, score)

            for skill in skills:
                self.habit_tracker.record_usage(
                    input_text=action_context,
                    skill_used=skill,
                    evaluation_score=score,
                    user_id=user_id
                )
            logging.info(f"RLSignalProcessor: Positive signal received (p_shift={p_shift:.2f}). Reinforced skills: {skills}")
        elif p_shift < 0.0:
            logging.info(f"RLSignalProcessor: Negative signal received (p_shift={p_shift:.2f}). No usage recorded.")
