import re
from typing import Dict, Any, Optional

class InteractiveFeedbackFormatter:
    """
    Formatter to extract explicit and implicit feedback signals from user text.
    Inspired by OpenClaw-RL interactive learning.
    """
    def __init__(self) -> None:
        """
        Initializes the InteractiveFeedbackFormatter.
        """
        pass

    def parse_corrections(self, text: str) -> Dict[str, Any]:
        """
        Parses user text to extract corrections and feedback signals.

        Args:
            text (str): The user's input text to be parsed.

        Returns:
            Dict[str, Any]: A dictionary containing feedback signals:
                - is_correction (bool): True if a correction was detected.
                - explicit_score (Optional[float]): Extracted explicit score (0-10), if any.
                - implicit_sentiment (float): Calculated implicit sentiment (-1.0 to 1.0).
        """
        if not text:
            return {
                "is_correction": False,
                "explicit_score": None,
                "implicit_sentiment": 0.0
            }

        text_lower = text.lower()

        # Detect corrections
        correction_keywords = ["no", "actually", "instead", "wrong", "fix", "incorrect"]
        is_correction = any(re.search(rf'\b{kw}\b', text_lower) for kw in correction_keywords)

        # Extract explicit score (e.g., 8/10 or 8.5 / 10)
        explicit_score: Optional[float] = None
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text_lower)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0 <= score <= 10:
                    explicit_score = score
            except ValueError:
                pass

        # Calculate implicit sentiment based on keywords
        positive_keywords = ["good", "great", "awesome", "perfect", "thanks", "correct", "yes"]
        negative_keywords = ["bad", "terrible", "awful", "incorrect", "wrong", "no", "stupid"]

        pos_count = sum(1 for kw in positive_keywords if re.search(rf'\b{kw}\b', text_lower))
        neg_count = sum(1 for kw in negative_keywords if re.search(rf'\b{kw}\b', text_lower))

        sentiment = float(pos_count - neg_count) * 0.5
        implicit_sentiment = max(min(sentiment, 1.0), -1.0)

        return {
            "is_correction": is_correction,
            "explicit_score": explicit_score,
            "implicit_sentiment": implicit_sentiment
        }
