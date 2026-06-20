import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class DialogueOnlineLearnerV3:
    """
    Online learning from dialogue v3.
    Extracts immediate context from conversation and dynamically adjusts agent behavior.
    """
    def __init__(self) -> None:
        self.active_modifiers: List[Dict[str, Any]] = []

    def process_turn(self, user_message: str, agent_response: Optional[str] = None) -> None:
        """
        Analyzes user message to extract immediate learning insights (preferences/constraints).
        """
        msg_lower = user_message.lower()

        # Very basic keyword heuristics to immediately adjust context
        if "always" in msg_lower or "must" in msg_lower:
            insight = {
                "type": "preference",
                "trigger": user_message,
                "effect": f"The user strongly prefers: '{user_message}'"
            }
            self.active_modifiers.append(insight)
            logger.info(f"Learned preference from dialogue: {insight['effect']}")

        elif "never" in msg_lower or "don't" in msg_lower or "do not" in msg_lower:
            insight = {
                "type": "constraint",
                "trigger": user_message,
                "effect": f"Constraint established: '{user_message}'"
            }
            self.active_modifiers.append(insight)
            logger.info(f"Learned constraint from dialogue: {insight['effect']}")

    def get_context_modifiers(self) -> str:
        """
        Returns a formatted string of active behavioral modifiers to be injected into prompts.
        """
        if not self.active_modifiers:
            return ""

        lines = ["--- Immediate Context Modifications ---"]
        for mod in self.active_modifiers:
            lines.append(f"- {mod['effect']}")
        return "\n".join(lines)

    def clear_session(self) -> None:
        """
        Clears the current dialogue modifiers for a new session.
        """
        self.active_modifiers.clear()
        logger.info("DialogueOnlineLearnerV3 session cleared.")
