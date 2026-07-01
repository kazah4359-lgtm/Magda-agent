import json
import logging
from typing import Dict, Any, Optional
from magda_agent.consciousness.core import Consciousness

logger = logging.getLogger(__name__)

class CanvasVisualizerV3:
    """
    Formats Magda's internal state into a structured JSON dictionary
    suitable for streaming to a live OpenClaw-inspired Canvas UI.
    V3 handles complex agent state including expanded memory systems,
    detailed planner states, emotions, and drives.
    """

    def __init__(self, consciousness: Consciousness) -> None:
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
        state: Dict[str, Any] = {
            "emotions": {},
            "mental_states": {},
            "memory": {
                "working": {},
                "episodic": {},
                "semantic": {},
                "procedural": {}
            },
            "skills": [],
            "planner": {},
            "drives": {},
            "global_workspace": {}
        }

        try:
            # 1. Emotions (Mirror Neurons, Insula, PAD)
            if self.consciousness.emotions:
                state["emotions"]["summary"] = self.consciousness.emotions.get_summary(user_id=user_id)

                history = self.consciousness.emotions.get_state_history(user_id=user_id)
                if history and len(history) > 0:
                    pad_state = history[0]
                    state["emotions"]["pad"] = {
                        "pleasure": getattr(pad_state, 'pleasure', 0.0),
                        "arousal": getattr(pad_state, 'arousal', 0.0),
                        "dominance": getattr(pad_state, 'dominance', 0.0)
                    }

            # 2. Mental States
            if self.consciousness.mental_states:
                state["mental_states"]["summary"] = self.consciousness.mental_states.get_summary(user_id=user_id)

            # 3. Memory Systems
            if self.consciousness.memory:
                memory_system = self.consciousness.memory

                # Working memory
                if hasattr(memory_system, 'working_memory') and memory_system.working_memory:
                    state["memory"]["working"] = {
                        "count": len(getattr(memory_system.working_memory, 'items', [])),
                        "summary": memory_system.working_memory.get_summary(user_id=user_id) if hasattr(memory_system.working_memory, 'get_summary') else None
                    }

                # Episodic memory
                if hasattr(memory_system, 'episodic') and memory_system.episodic:
                    state["memory"]["episodic"] = {
                        "status": "active"
                    }
                elif hasattr(self.consciousness, 'long_term_memory') and self.consciousness.long_term_memory:
                     state["memory"]["episodic"] = {
                        "status": "legacy_active"
                    }

                # Semantic memory
                if hasattr(memory_system, 'semantic') and memory_system.semantic:
                    state["memory"]["semantic"] = {
                        "status": "active"
                    }

                # Procedural memory
                if hasattr(memory_system, 'procedural') and memory_system.procedural:
                    state["memory"]["procedural"] = {
                        "status": "active",
                        "skills_count": len(getattr(memory_system.procedural, 'skills', {})) if hasattr(memory_system.procedural, 'skills') else 0
                    }

            # 4. Skills
            if self.consciousness.skills:
                skills_dict = getattr(self.consciousness.skills, 'skills', {})
                state["skills"] = list(skills_dict.keys())

            # 5. Planner
            if self.consciousness.planner:
                planner = self.consciousness.planner
                state["planner"]["summary"] = planner.get_state_summary() if hasattr(planner, 'get_state_summary') else None

                if hasattr(planner, 'current_plan') and planner.current_plan:
                    plan = planner.current_plan
                    state["planner"]["current_plan"] = {
                        "goal": getattr(plan, 'goal', None),
                        "step_count": len(getattr(plan, 'steps', [])),
                        "dependencies": getattr(plan, 'dependencies', [])
                    }

            # 6. Drives (Hypothalamus)
            if self.consciousness.hypothalamus:
                hypothalamus = self.consciousness.hypothalamus
                state["drives"] = {
                    "energy": getattr(hypothalamus, 'energy', 1.0),
                    "boredom": getattr(hypothalamus, 'boredom', 0.0),
                    "summary": hypothalamus.get_summary() if hasattr(hypothalamus, 'get_summary') else None
                }

            # 7. Global Workspace
            if hasattr(self.consciousness, 'global_workspace') and self.consciousness.global_workspace:
                workspace = self.consciousness.global_workspace
                state["global_workspace"] = {
                    "focused_event": getattr(workspace, 'focused_event', None),
                    "active": True
                }

        except Exception as e:
            logger.error(f"Error formatting canvas v3 state: {e}")
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
