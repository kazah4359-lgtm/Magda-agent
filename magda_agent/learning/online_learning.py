import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class OnlineLearnerModule:
    """
    Extracts insights from dialogue in real time to enable online learning.
    """
    def __init__(self, db_connection: Any = None):
        self.db = db_connection
        self.insights: List[Dict[str, Any]] = []

    def process_dialogue(self, user_message: str, agent_response: str) -> None:
        """
        Processes a dialogue turn to extract immediate insights.
        """
        # Very simple heuristic: if user says "don't" or "never", it's a constraint insight.
        if "don't" in user_message.lower() or "never" in user_message.lower():
            insight = {
                "type": "constraint",
                "content": user_message,
                "context_response": agent_response
            }
            self.insights.append(insight)
            logger.info(f"Learned constraint insight: {insight}")

        # If user says "always" or "must", it's a preference insight.
        elif "always" in user_message.lower() or "must" in user_message.lower():
            insight = {
                "type": "preference",
                "content": user_message,
                "context_response": agent_response
            }
            self.insights.append(insight)
            logger.info(f"Learned preference insight: {insight}")

    def get_recent_insights(self) -> List[Dict[str, Any]]:
        """
        Returns the recently extracted insights.
        """
        return self.insights

    def clear_insights(self) -> None:
        self.insights = []
