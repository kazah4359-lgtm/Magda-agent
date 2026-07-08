import logging
from typing import Optional, Dict, Any
from magda_agent.metacognition.tracker import QualityTracker
from magda_agent.learning.interactive_feedback import InteractiveFeedbackFormatterV2
from magda_agent.llm_client import LLMClient

class RLMetricsSystemV2:
    """
    OpenClaw RL Metrics System v2.
    Captures longitudinal quality updates based on next-state signals (user replies).
    Uses the V2 feedback formatter to extract more detailed intent.
    Optionally, uses the LLM to understand deeper nuances if simple parsing yields a neutral intent.
    """
    def __init__(self, quality_tracker: QualityTracker, llm_client: Optional[LLMClient] = None) -> None:
        """
        Initializes the RL Metrics System v2.

        Args:
            quality_tracker (QualityTracker): Tracker to log metrics longitudinally.
            llm_client (Optional[LLMClient]): The LLM client to use for complex feedback analysis.
        """
        self.quality_tracker = quality_tracker
        self.formatter = InteractiveFeedbackFormatterV2()
        self.llm_client = llm_client

    async def capture_next_state_signal(
        self,
        user_reply: str,
        action_context: str,
        user_id: Optional[str],
        tool_output: Optional[str] = None
    ) -> None:
        """
        Analyzes the user's reply as a next-state signal and logs RL metrics.
        Uses LLM for deeper analysis if the simple parser is unsure.

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

        feedback_signals = self.formatter.parse_detailed_feedback(signal_text)

        intent = feedback_signals.get("intent", "neutral")
        explicit_score = feedback_signals.get("explicit_score")
        implicit_sentiment = feedback_signals.get("implicit_sentiment", 0.0)
        is_correction = feedback_signals.get("is_correction", False)

        # Fallback to LLM if the simple parser couldn't confidently classify intent
        if intent == "neutral" and self.llm_client is not None:
            prompt = f"Analyze the following user feedback in the context of action: {action_context}\nFeedback: '{user_reply}'\nIs this feedback a 'correction', 'praise', or 'neutral'?"
            try:
                llm_response = await self.llm_client.generate(prompt)
                llm_text = llm_response.lower()
                if "correction" in llm_text or "criticism" in llm_text:
                    intent = "criticism"
                    is_correction = True
                    implicit_sentiment = min(implicit_sentiment, -0.5)
                elif "praise" in llm_text:
                    intent = "praise"
                    implicit_sentiment = max(implicit_sentiment, 0.5)
            except Exception as e:
                logging.error(f"RLMetricsSystemV2: LLM fallback failed: {e}")

        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "action_context": action_context,
            "is_correction": is_correction,
            "intent": intent
        }

        # Log explicit score if available
        if explicit_score is not None:
            self.quality_tracker.log_metric("rl_explicit_score_v2", explicit_score, metadata)
            logging.info(f"RLMetricsSystemV2: Logged rl_explicit_score_v2={explicit_score} for user {user_id}")

        # Log implicit sentiment
        self.quality_tracker.log_metric("rl_implicit_sentiment_v2", implicit_sentiment, metadata)
        logging.info(f"RLMetricsSystemV2: Logged rl_implicit_sentiment_v2={implicit_sentiment} for user {user_id}")

        # Calculate a combined quality score (0-100)
        if explicit_score is not None:
            quality_score = explicit_score * 10
        else:
            # Map sentiment [-1.0, 1.0] to [0, 100], base 50
            quality_score = 50 + (implicit_sentiment * 50)
            if is_correction or intent == "criticism":
                quality_score = max(0.0, quality_score - 20)
            elif intent == "praise":
                quality_score = min(100.0, quality_score + 10)

        self.quality_tracker.log_metric("rl_quality_score_v2", quality_score, metadata)
        logging.info(f"RLMetricsSystemV2: Logged rl_quality_score_v2={quality_score} for user {user_id} with intent {intent}")
