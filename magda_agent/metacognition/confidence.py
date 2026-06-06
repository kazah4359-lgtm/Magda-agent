import logging
from typing import Optional
from magda_agent.llm_client import LLMClient
from magda_agent.metacognition.tracker import QualityTracker

class ConfidenceCalibrator:
    """
    Estimates confidence in the agent's responses and tracks calibration accuracy.
    """
    def __init__(self, llm: LLMClient, tracker: QualityTracker) -> None:
        self.llm = llm
        self.tracker = tracker
        self.last_confidence: Optional[float] = None

    async def estimate_confidence(self, user_input: str, response: str) -> float:
        """
        Estimates confidence score between 0.0 and 1.0 based on user input and agent response.
        """
        prompt = (
            "Evaluate your confidence in the following response to the user's input.\n"
            "Return a single float value between 0.0 and 1.0 representing your confidence.\n"
            "High confidence (e.g., 0.9) means you are certain of the facts and logic.\n"
            "Low confidence (e.g., 0.3) means you are guessing or lack information.\n\n"
            f"User input: {user_input}\n"
            f"Response: {response}\n\n"
            "Respond ONLY with a number between 0.0 and 1.0."
        )
        messages = [{"role": "system", "content": prompt}]
        try:
            result_text = await self.llm.chat_completion(messages, temperature=0.1)
            score = float(result_text.strip())
            score = max(0.0, min(1.0, score))
            self.last_confidence = score
            return score
        except Exception as e:
            logging.error(f"Failed to estimate confidence: {e}")
            self.last_confidence = 0.5
            return 0.5

    def add_caveat_if_needed(self, response: str, confidence: float, threshold: float = 0.6) -> str:
        """
        Appends a caveat if confidence is below the threshold.
        """
        if confidence < threshold:
            caveat = "\n\n*(Note: I am not completely confident in this answer, please verify the details.)*"
            return response + caveat
        return response

    def track_calibration(self, confidence: float, actual_score: float) -> None:
        """
        Tracks the difference between predicted confidence (0-1) and actual score (0-10) scaled to (0-1).
        """
        actual_scaled = actual_score / 10.0
        calibration_error = abs(confidence - actual_scaled)
        self.tracker.log_metric(
            "calibration_error",
            calibration_error,
            metadata={"confidence": confidence, "actual_score": actual_score}
        )
