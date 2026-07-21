import logging
from typing import Any, Dict, List, Optional
from magda_agent.emotions.mirror_neurons import MirrorNeurons

logger = logging.getLogger(__name__)

class DialogueOnlineLearnerV4:
    """
    Online learning from dialogue v4.
    Extracts immediate context from conversation and dynamically adjusts agent behavior.
    Integrates learning loop into conversation processor to capture state-action-reward from next-state responses.
    Inspired by OpenClaw-RL trend (June 2026).
    """
    def __init__(self, mirror_neurons: Optional[MirrorNeurons] = None) -> None:
        self.mirror_neurons = mirror_neurons or MirrorNeurons()
        self.active_modifiers: List[Dict[str, Any]] = []
        # Initial behavior weights
        self.weights = {
            "verbosity": 1.0,
            "directness": 1.0,
            "empathy": 1.0
        }
        self.last_state_action: Optional[Dict[str, str]] = None
        self.trajectory_log: List[Dict[str, Any]] = []

    def capture_state_action(self, state_context: str, action_response: str) -> None:
        """
        Captures the current state and action for the next-state reward calculation.
        """
        self.last_state_action = {
            "state": state_context,
            "action": action_response
        }
        logger.info("Captured state-action pair for next-state learning loop.")

    def process_turn(self, user_message: str, agent_response: Optional[str] = None) -> None:
        """
        Analyzes user message to extract immediate learning insights and adjust behavior weights.
        Processes state-action-reward if a previous state-action exists.
        """
        msg_lower = user_message.lower()

        # 1. Direct instructions/preferences extraction
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

        # 2. Online RL from sentiment signals (MirrorNeurons) as Next-State Reward
        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_message)

        # State-Action-Reward logging
        if self.last_state_action:
            reward = p_shift
            trajectory_entry = {
                "state": self.last_state_action["state"],
                "action": self.last_state_action["action"],
                "next_state": user_message,
                "reward": reward
            }
            self.trajectory_log.append(trajectory_entry)
            logger.info(f"Logged trajectory with reward {reward:.2f}.")
            self.last_state_action = None

        # Adjust weights based on pleasure shift (P-shift) as a reward signal
        if p_shift > 0:
            # Positive reinforcement
            self.weights["verbosity"] = min(2.0, self.weights["verbosity"] + 0.1)
            self.weights["empathy"] = min(2.0, self.weights["empathy"] + 0.1)
            logger.info(f"Positive dialogue signal (P={p_shift:.2f}). Reinforced verbosity and empathy.")
        elif p_shift < 0:
            # Negative reinforcement
            self.weights["verbosity"] = max(0.5, self.weights["verbosity"] - 0.1)
            self.weights["directness"] = min(2.0, self.weights["directness"] + 0.1)
            logger.info(f"Negative dialogue signal (P={p_shift:.2f}). Reduced verbosity, increased directness.")

    def get_context_modifiers(self) -> str:
        """
        Returns a formatted string of active behavioral modifiers and weights to be injected into prompts.
        """
        sections = []

        if self.active_modifiers:
            lines = ["--- Immediate Context Modifications ---"]
            for mod in self.active_modifiers:
                lines.append(f"- {mod['effect']}")
            sections.append("\n".join(lines))

        # Behavior weights section
        weight_lines = ["--- Dynamic Behavior Adjustments ---"]
        for key, value in self.weights.items():
            if value > 1.2:
                weight_lines.append(f"- Increase {key} (Strength: {value:.1f})")
            elif value < 0.8:
                weight_lines.append(f"- Decrease {key} (Strength: {value:.1f})")

        if len(weight_lines) > 1:
            sections.append("\n".join(weight_lines))

        return "\n\n".join(sections) if sections else ""

    def clear_session(self) -> None:
        """
        Clears the current dialogue modifiers and resets weights for a new session.
        """
        self.active_modifiers.clear()
        self.weights = {"verbosity": 1.0, "directness": 1.0, "empathy": 1.0}
        self.last_state_action = None
        self.trajectory_log.clear()
        logger.info("DialogueOnlineLearnerV4 session cleared.")
