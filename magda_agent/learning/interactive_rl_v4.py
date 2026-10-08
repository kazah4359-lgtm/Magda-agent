import logging
import re
from typing import Dict, Optional, List, Any

from magda_agent.llm_client import LLMClient
from magda_agent.learning.habits import HabitTracker
from magda_agent.user_model.model import UserModel


class InteractiveRLLoopV4:
    """
    OpenClaw-RL Interactive Learning Loop V4.
    Evaluates next-state signals (user replies, tool outputs) using LLM reasoning
    (with heuristic fallback) to adjust skill weights, record habit usage, and update user models.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        habit_tracker: Optional[HabitTracker] = None,
        user_model: Optional[UserModel] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.2,
    ) -> None:
        """
        Initializes InteractiveRLLoopV4.

        Args:
            llm_client (Optional[LLMClient]): LLM client for evaluating user replies.
            habit_tracker (Optional[HabitTracker]): HabitTracker instance to record successful interactions.
            user_model (Optional[UserModel]): UserModel instance to persist user preferences.
            initial_weights (Optional[Dict[str, float]]): Initial skill weights dictionary.
            learning_rate (float): Learning rate adjustment factor for skill weights.
        """
        self.llm_client = llm_client
        self.habit_tracker = habit_tracker
        self.user_model = user_model
        self.learning_rate = learning_rate
        self.learning_state: Dict[str, float] = dict(initial_weights) if initial_weights else {}
        logging.info("Initialized InteractiveRLLoopV4")

    def analyze_feedback_heuristic(
        self,
        reply_text: str,
        tool_output: Optional[str] = None,
    ) -> float:
        """
        Calculates a heuristic reward score from a user reply and optional tool output.

        Args:
            reply_text (str): User reply text.
            tool_output (Optional[str]): Tool execution output.

        Returns:
            float: Score between 0.0 and 10.0.
        """
        reply_lower = (reply_text or "").lower()

        negative_words = [r"\bno\b", r"\bwrong\b", r"\bbad\b", r"\bincorrect\b", r"\bterrible\b", r"\bstop\b", r"\bfail\b"]
        positive_words = [r"\bthanks\b", r"\bthank you\b", r"\bgreat\b", r"\bawesome\b", r"\bperfect\b", r"\bgood\b", r"\byes\b", r"\bcorrect\b"]

        if any(re.search(word, reply_lower) for word in negative_words):
            score = 2.0
        elif any(re.search(word, reply_lower) for word in positive_words):
            score = 8.5
        else:
            score = 5.0

        if tool_output and score >= 5.0:
            score = min(10.0, score + 1.0)

        return score

    async def analyze_feedback_llm(
        self,
        reply_text: str,
        action_context: str,
        tool_output: Optional[str] = None,
    ) -> float:
        """
        Evaluates feedback using an LLM prompt with fallbacks to heuristic scoring.

        Args:
            reply_text (str): User reply evaluating the interaction.
            action_context (str): Context of the action performed.
            tool_output (Optional[str]): Output from tool execution if available.

        Returns:
            float: Calculated reward scalar between 0.0 and 10.0.
        """
        if not self.llm_client:
            return self.analyze_feedback_heuristic(reply_text, tool_output)

        prompt = (
            f"You are an RL evaluator analyzing next-state signals.\n"
            f"Action Context: {action_context}\n"
            f"Tool Output: {tool_output or 'N/A'}\n"
            f"User Reply: {reply_text}\n"
            f"Rate the user's satisfaction and success of the action on a scale from 0.0 to 10.0.\n"
            f"Return ONLY a single float number between 0.0 and 10.0."
        )

        try:
            response = await self.llm_client.generate(prompt, temperature=0.0)
            match = re.search(r"(\d+(?:\.\d+)?)", response)
            if match:
                val = float(match.group(1))
                return max(0.0, min(10.0, val))
        except Exception as e:
            logging.warning(f"InteractiveRLLoopV4 LLM evaluation failed: {e}. Using heuristic fallback.")

        return self.analyze_feedback_heuristic(reply_text, tool_output)

    async def process_interaction(
        self,
        user_reply: str,
        action_context: str,
        skill_used: str = "default_skill",
        tool_output: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> float:
        """
        Processes a single interaction turn, updating skill weights, habit tracker, and user model.

        Args:
            user_reply (str): Next-state user reply signal.
            action_context (str): Action context description.
            skill_used (str): Name of the skill executed.
            tool_output (Optional[str]): Tool execution output.
            user_id (Optional[int]): User ID.

        Returns:
            float: Evaluated reward score.
        """
        if not user_reply and not tool_output:
            return 5.0

        reward = await self.analyze_feedback_llm(
            reply_text=user_reply,
            action_context=action_context,
            tool_output=tool_output,
        )

        # Update skill weight
        current_weight = self.learning_state.get(skill_used, 1.0)
        # Shift relative to neutral baseline of 5.0
        weight_delta = (reward - 5.0) / 5.0 * self.learning_rate
        new_weight = max(0.1, min(10.0, current_weight + weight_delta))
        self.learning_state[skill_used] = new_weight

        # Record habit usage if positive signal
        if self.habit_tracker and reward >= 5.0:
            self.habit_tracker.record_usage(
                input_text=action_context,
                skill_used=skill_used,
                evaluation_score=reward,
                user_id=user_id,
            )

        # Update user model if available
        if self.user_model and user_id is not None:
            model_data = self.user_model.get_model(user_id)
            if "rl_v4_history" not in model_data:
                model_data["rl_v4_history"] = []

            model_data["rl_v4_history"].append(
                {
                    "skill": skill_used,
                    "reward": reward,
                    "weight": new_weight,
                }
            )

            if reward >= 7.0:
                model_data["communication_style"] = (
                    f"{model_data.get('communication_style', 'default')} (reinforced)"
                )

            self.user_model.save_model(user_id, model_data)

        logging.info(
            f"InteractiveRLLoopV4: Processed interaction for skill '{skill_used}'. "
            f"Reward: {reward:.2f}, New Weight: {new_weight:.2f}"
        )
        return reward

    def get_state(self) -> Dict[str, float]:
        """
        Retrieves the current learning state (skill weights).

        Returns:
            Dict[str, float]: Dictionary mapping skill names to weights.
        """
        return dict(self.learning_state)


OpenClawInteractiveLearnerV4 = InteractiveRLLoopV4
