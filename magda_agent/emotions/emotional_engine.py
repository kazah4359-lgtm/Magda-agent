from typing import Dict

class EmotionalEngine:
    def __init__(self):
        # PAD model: Pleasure, Arousal, Dominance
        self.state = {
            "pleasure": 0.0,  # Positive vs Negative
            "arousal": 0.0,   # Excited vs Calm
            "dominance": 0.0  # Controlled vs Controlling
        }
        self.decay_rate = 0.1

    def update(self, event_impact: Dict[str, float]):
        """
        Updates the emotional state based on an event.
        event_impact: {"pleasure": 0.5, "arousal": 0.8, "dominance": -0.2}
        """
        for key in self.state:
            if key in event_impact:
                self.state[key] += event_impact[key]
                # Clamp between -1.0 and 1.0
                self.state[key] = max(-1.0, min(1.0, self.state[key]))

    def decay(self):
        """
        Emotions naturally return to baseline (0.0) over time.
        """
        for key in self.state:
            if self.state[key] > 0:
                self.state[key] = max(0, self.state[key] - self.decay_rate)
            elif self.state[key] < 0:
                self.state[key] = min(0, self.state[key] + self.decay_rate)

    def get_state(self) -> Dict[str, float]:
        return self.state.copy()

    def get_mood_label(self) -> str:
        p, a, d = self.state["pleasure"], self.state["arousal"], self.state["dominance"]
        if p > 0.5 and a > 0.5: return "Exuberant"
        if p > 0.5 and a < -0.5: return "Relaxed"
        if p < -0.5 and a > 0.5: return "Anxious/Angry"
        if p < -0.5 and a < -0.5: return "Bored/Depressed"
        if p > 0: return "Content"
        if p < 0: return "Displeased"
        return "Neutral"
