import json
import logging
from typing import Dict, Any, Optional
from magda_agent.consciousness.core import Consciousness

logger = logging.getLogger(__name__)

class CanvasVisualizer:
    """
    Formats Magda's internal state into a structured JSON dictionary
    suitable for streaming to a live OpenClaw-inspired Canvas UI.
    """

    def __init__(self, consciousness: Consciousness):
        """
        Initializes the visualizer with a reference to the agent's consciousness.

        Args:
            consciousness: The main Consciousness instance containing the agent's state.
        """
        self.consciousness = consciousness

    def get_formatted_state(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves and formats the agent's internal state into a structured dictionary.

        Args:
            user_id: Optional user identifier to filter state (e.g., for emotions/memory).

        Returns:
            Dict[str, Any]: A dictionary representing the agent's current state.
        """
        state = {
            "emotions": {},
            "mental_states": {},
            "memory": {},
            "skills": [],
            "planner": {}
        }

        try:
            # 1. Emotions
            if self.consciousness.emotions:
                state["emotions"] = {
                    "summary": self.consciousness.emotions.get_summary(user_id=user_id)
                }

                # Try to get raw PAD values if available
                history = self.consciousness.emotions.get_state_history(user_id=user_id)
                if history and len(history) > 0:
                    pad_state = history[0]
                    # Assuming PADState has pleasure, arousal, dominance attributes
                    state["emotions"]["pad"] = {
                        "pleasure": getattr(pad_state, 'pleasure', 0.0),
                        "arousal": getattr(pad_state, 'arousal', 0.0),
                        "dominance": getattr(pad_state, 'dominance', 0.0)
                    }

            # 2. Mental States
            if self.consciousness.mental_states:
                state["mental_states"] = {
                    "summary": self.consciousness.mental_states.get_summary(user_id=user_id)
                }

            # 3. Memory
            if self.consciousness.memory:
                state["memory"] = {
                    "summary": self.consciousness.memory.get_summary()
                }

            # 4. Skills
            if self.consciousness.skills:
                # Extract skill names from the registry
                skills_dict = getattr(self.consciousness.skills, 'skills', {})
                state["skills"] = list(skills_dict.keys())

            # 5. Planner
            if self.consciousness.planner:
                planner_state = self.consciousness.planner.get_state_summary()
                state["planner"] = {
                    "summary": planner_state
                }

        except Exception as e:
            logger.error(f"Error formatting canvas state: {e}")
            state["error"] = str(e)

        return state

    def get_state_json(self, user_id: Optional[str] = None) -> str:
        """
        Returns the formatted state as a JSON string.

        Args:
            user_id: Optional user identifier.

        Returns:
            str: JSON-encoded string of the agent state.
        """
        return json.dumps(self.get_formatted_state(user_id=user_id))
