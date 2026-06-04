import time
from typing import Dict

class Hypothalamus:
    """
    Module responsible for the agent's basic needs (homeostasis).
    Tracks and updates levels for different needs such as social, rest, and curiosity.
    Need values range from 0.0 (fully satisfied) to 1.0 (extreme need).
    """

    def __init__(self) -> None:
        """
        Initializes the agent's needs with default values.
        """
        self.needs: Dict[str, float] = {
            "social": 0.5,     # Need for interaction
            "rest": 0.0,       # Need for idle time/processing
            "curiosity": 0.5,  # Need for new information/skills
        }
        self.last_update_time: float = time.time()

    def update_needs(self, delta_social: float = 0.0, delta_rest: float = 0.0, delta_curiosity: float = 0.0) -> None:
        """
        Updates the agent's needs based on explicit deltas and the passage of time.
        Positive deltas increase the need, negative deltas satisfy (decrease) the need.

        Args:
            delta_social (float): Explicit change to the social need.
            delta_rest (float): Explicit change to the rest need.
            delta_curiosity (float): Explicit change to the curiosity need.
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time

        # Natural decay/growth over time (simplified)
        # Social need grows over time
        # Rest need grows slightly over time (simulating fatigue)
        # Curiosity grows slowly
        time_factor = min(time_elapsed / 3600.0, 1.0) # max 1 hour at a time

        self.needs["social"] = min(1.0, max(0.0, self.needs["social"] + delta_social + (time_factor * 0.1)))
        self.needs["rest"] = min(1.0, max(0.0, self.needs["rest"] + delta_rest + (time_factor * 0.05)))
        self.needs["curiosity"] = min(1.0, max(0.0, self.needs["curiosity"] + delta_curiosity + (time_factor * 0.05)))

        self.last_update_time = current_time

    def get_summary(self) -> str:
        """
        Returns a string summary of the agent's current needs for context injection.

        Returns:
            str: A formatted string describing the current state of needs.
        """
        return (
            f"Needs State: "
            f"Social: {self.needs['social']:.2f}, "
            f"Rest: {self.needs['rest']:.2f}, "
            f"Curiosity: {self.needs['curiosity']:.2f}"
        )
