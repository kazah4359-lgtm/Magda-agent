import logging
from typing import Optional, Dict, Any
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.learning.interactive_feedback import InteractiveFeedbackFormatter

class RLMetricsSystem:
    """
    OpenClaw RL Metrics System.
    Captures longitudinal quality updates based on next-state signals (user replies).
    """
    def __init__(self, quality_tracker: QualityTracker) -> None:
        """
        Initializes the RL Metrics System.

        Args:
            quality_tracker (QualityTracker): Tracker to log metrics longitudinally.
        """
        self.quality_tracker = quality_tracker
        self.formatter = InteractiveFeedbackFormatter()

    def capture_next_state_signal(
        self,
        user_reply: str,
        action_context: str,
        user_id: Optional[str],
        tool_output: Optional[str] = None
    ) -> None:
        """
        Analyzes the user's reply as a next-state signal and logs RL metrics.

        Args:
            user_reply (str): The user's reply text.
            action_context (str): The context of the action that was taken.
            user_id (Optional[str]): The user's ID.
            tool_output (Optional[str], optional): The output of the tool, if any. Defaults to None.
        """
        if not user_reply:
            return

        signal_text = user_reply
        if tool_output:
            signal_text += f" [Tool Output: {tool_output}]"

        feedback_signals = self.formatter.parse_corrections(signal_text)

        explicit_score = feedback_signals.get("explicit_score")
        implicit_sentiment = feedback_signals.get("implicit_sentiment", 0.0)
        is_correction = feedback_signals.get("is_correction", False)

        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "action_context": action_context,
            "is_correction": is_correction
        }

        # Log explicit score if available
        if explicit_score is not None:
            self.quality_tracker.log_metric("rl_explicit_score", explicit_score, metadata)
            logging.info(f"RLMetricsSystem: Logged rl_explicit_score={explicit_score} for user {user_id}")

        # Log implicit sentiment
        self.quality_tracker.log_metric("rl_implicit_sentiment", implicit_sentiment, metadata)
        logging.info(f"RLMetricsSystem: Logged rl_implicit_sentiment={implicit_sentiment} for user {user_id}")

        # Calculate a combined quality score (0-100)
        if explicit_score is not None:
            quality_score = explicit_score * 10
        else:
            # Map sentiment [-1.0, 1.0] to [0, 100], base 50
            quality_score = 50 + (implicit_sentiment * 50)
            if is_correction:
                quality_score = max(0.0, quality_score - 20)

        self.quality_tracker.log_metric("rl_quality_score", quality_score, metadata)
        logging.info(f"RLMetricsSystem: Logged rl_quality_score={quality_score} for user {user_id}")
