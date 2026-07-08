import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MemoryCanvasVisualizer:
    """
    Formats Magda's internal memory state into a structured JSON dictionary
    suitable for streaming to a live OpenClaw-inspired Canvas UI.
    """

    def __init__(self, memory_system: Any) -> None:
        """
        Initializes the visualizer with a reference to the agent's memory system.

        Args:
            memory_system: The memory system component containing working, episodic, semantic, and procedural memory.
        """
        self.memory_system = memory_system

    def get_formatted_memory_state(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves and formats the agent's memory state into a structured dictionary.

        Args:
            user_id: Optional user identifier to filter state (e.g., for memory retrieval).

        Returns:
            Dict[str, Any]: A dictionary representing the agent's current memory state.
        """
        state: Dict[str, Any] = {
            "working": {},
            "episodic": {},
            "semantic": {},
            "procedural": {}
        }

        if not self.memory_system:
            return state

        try:
            # Working memory
            if hasattr(self.memory_system, 'working_memory') and self.memory_system.working_memory:
                wm = self.memory_system.working_memory

                user_id_int = None
                if user_id is not None:
                    try:
                        user_id_int = int(user_id)
                    except ValueError:
                        pass

                entries = []
                if hasattr(wm, 'get_entries'):
                    entries = wm.get_entries(user_id=user_id_int)
                elif hasattr(wm, 'items'):
                    entries = getattr(wm, 'items', [])

                count = len(entries)
                summary = wm.get_summary(user_id=user_id_int) if hasattr(wm, 'get_summary') else None

                state["working"] = {
                    "count": count,
                    "summary": summary
                }

            # Episodic memory
            if hasattr(self.memory_system, 'episodic') and self.memory_system.episodic:
                state["episodic"] = {
                    "status": "active"
                }

            # Semantic memory
            if hasattr(self.memory_system, 'semantic') and self.memory_system.semantic:
                state["semantic"] = {
                    "status": "active"
                }

            # Procedural memory
            if hasattr(self.memory_system, 'procedural') and self.memory_system.procedural:
                proc = self.memory_system.procedural
                skills_count = len(getattr(proc, 'skills', {})) if hasattr(proc, 'skills') else 0
                state["procedural"] = {
                    "status": "active",
                    "skills_count": skills_count
                }

        except Exception as e:
            logger.error(f"Error formatting memory canvas state: {e}")
            state["error"] = str(e)

        return state

    def get_memory_state_json(self, user_id: Optional[str] = None) -> str:
        """
        Returns the formatted memory state as a JSON string.

        Args:
            user_id: Optional user identifier.

        Returns:
            str: JSON-encoded string of the agent memory state.
        """
        return json.dumps(self.get_formatted_memory_state(user_id=user_id))
