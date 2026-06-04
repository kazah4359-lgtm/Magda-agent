import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class PADState:
    pleasure: float = 0.0  # -1.0 to 1.0 (Displeasure to Pleasure)
    arousal: float = 0.0   # -1.0 to 1.0 (Non-arousal to Arousal)
    dominance: float = 0.0 # -1.0 to 1.0 (Submissiveness to Dominance)

    def to_dict(self) -> Dict[str, float]:
        return {
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance
        }

class EmotionalEngine:
    """
    Mathematical Emotional Engine based on the PAD model.
    Allows for continuous emotional state representation and updates.
    Supports per-user emotional states to prevent cross-user contamination.
    """
    HISTORY_MAXLEN = 500

    def __init__(self, decay_rate: float = 0.05):
        self._user_states: Dict[Optional[int], PADState] = {None: PADState()}
        self.decay_rate = decay_rate
        self.history: deque = deque(maxlen=self.HISTORY_MAXLEN)

    def _get_state(self, user_id: Optional[int] = None) -> PADState:
        """Get or create a PAD state for a specific user."""
        if user_id not in self._user_states:
            self._user_states[user_id] = PADState()
        return self._user_states[user_id]

    @property
    def state(self) -> PADState:
        """Default state (user_id=None) for backward compatibility."""
        return self._get_state(None)

    @state.setter
    def state(self, value: PADState):
        self._user_states[None] = value

    def update(self, p_delta: float, a_delta: float, d_delta: float, user_id: Optional[int] = None):
        """Update the emotional state with new stimuli."""
        state = self._get_state(user_id)
        state.pleasure = self._clamp(state.pleasure + p_delta)
        state.arousal = self._clamp(state.arousal + a_delta)
        state.dominance = self._clamp(state.dominance + d_delta)
        self.history.append(PADState(state.pleasure, state.arousal, state.dominance))

    def decay(self, user_id: Optional[int] = None):
        """Gradually return to neutral state (0,0,0)."""
        state = self._get_state(user_id)
        state.pleasure *= (1 - self.decay_rate)
        state.arousal *= (1 - self.decay_rate)
        state.dominance *= (1 - self.decay_rate)

    def get_emotion_label(self, user_id: Optional[int] = None) -> str:
        """Map PAD values to basic human emotion labels."""
        state = self._get_state(user_id)
        p, a, d = state.pleasure, state.arousal, state.dominance

        if p > 0.3:
            if a > 0.3:
                return "Excited/Happy" if d > 0 else "Docile/Pleasant"
            elif a < -0.3:
                return "Relaxed/Calm"
            else:
                return "Content"
        elif p < -0.3:
            if a > 0.3:
                return "Angry/Hostile" if d > 0 else "Anxious/Fearful"
            elif a < -0.3:
                return "Bored/Sad"
            else:
                return "Displeased"
        else:
            if a > 0.5:
                return "Surprised"
            return "Neutral"

    def _clamp(self, value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        return max(min_val, min(max_val, value))

    def get_summary(self, user_id: Optional[int] = None) -> str:
        state = self._get_state(user_id)
        label = self.get_emotion_label(user_id)
        return f"Current Emotion: {label} (P:{state.pleasure:.2f}, A:{state.arousal:.2f}, D:{state.dominance:.2f})"

