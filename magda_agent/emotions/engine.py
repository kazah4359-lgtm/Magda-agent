import math
from dataclasses import dataclass, field
from typing import Dict, List

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
    """
    def __init__(self, decay_rate: float = 0.05):
        self.state = PADState()
        self.decay_rate = decay_rate
        self.history: List[PADState] = []

    def update(self, p_delta: float, a_delta: float, d_delta: float):
        """Update the emotional state with new stimuli."""
        self.state.pleasure = self._clamp(self.state.pleasure + p_delta)
        self.state.arousal = self._clamp(self.state.arousal + a_delta)
        self.state.dominance = self._clamp(self.state.dominance + d_delta)
        self.history.append(PADState(self.state.pleasure, self.state.arousal, self.state.dominance))

    def decay(self):
        """Gradually return to neutral state (0,0,0)."""
        self.state.pleasure *= (1 - self.decay_rate)
        self.state.arousal *= (1 - self.decay_rate)
        self.state.dominance *= (1 - self.decay_rate)

    def get_emotion_label(self) -> str:
        """Map PAD values to basic human emotion labels."""
        p, a, d = self.state.pleasure, self.state.arousal, self.state.dominance

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

    def get_summary(self) -> str:
        return f"Current Emotion: {self.get_emotion_label()} (P:{self.state.pleasure:.2f}, A:{self.state.arousal:.2f}, D:{self.state.dominance:.2f})"
