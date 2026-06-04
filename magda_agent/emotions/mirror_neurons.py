import re
from typing import Tuple

class MirrorNeurons:
    """
    Mirror Neuron System module.
    Responsible for basic empathy by analyzing user text for positive/negative words
    and returning a shift for PAD (Pleasure, Arousal, Dominance) to mirror the user's emotional state.
    """

    def __init__(self) -> None:
        """Initializes the Mirror Neuron System."""
        self.positive_words = {
            "good", "happy", "great", "excellent", "joy", "love", "awesome",
            "fantastic", "wonderful", "glad", "excited", "amazing", "beautiful"
        }
        self.negative_words = {
            "bad", "sad", "angry", "terrible", "awful", "hate", "depressed",
            "upset", "miserable", "annoyed", "frustrated", "pain", "hurt"
        }

    def empathize(self, text: str) -> Tuple[float, float, float]:
        """
        Analyzes the text for emotional tone and returns a PAD shift to empathize.

        Args:
            text (str): The user's input text.

        Returns:
            Tuple[float, float, float]: The shift for (pleasure, arousal, dominance).
        """
        if not text:
            return (0.0, 0.0, 0.0)

        words = re.findall(r'\b\w+\b', text.lower())

        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)

        if pos_count > neg_count:
            # Empathize with positive emotion
            p_shift = min(0.1 * (pos_count - neg_count), 0.5)
            a_shift = min(0.05 * (pos_count - neg_count), 0.3)
            d_shift = 0.0
            return (p_shift, a_shift, d_shift)
        elif neg_count > pos_count:
            # Empathize with negative emotion (lower pleasure, slightly increase arousal and dominance to offer help)
            p_shift = max(-0.1 * (neg_count - pos_count), -0.5)
            a_shift = min(0.05 * (neg_count - pos_count), 0.3)
            d_shift = min(0.02 * (neg_count - pos_count), 0.2)
            return (p_shift, a_shift, d_shift)

        return (0.0, 0.0, 0.0)
