from typing import Dict

class Hypothalamus:
    """
    Hypothalamus module.
    Responsible for managing homeostatic drives such as energy and boredom.
    These drives can influence the agent's internal state and behavior.
    """

    def __init__(self, initial_energy: float = 1.0, initial_boredom: float = 0.0) -> None:
        """
        Initializes the Hypothalamus module with starting drive levels.

        Args:
            initial_energy (float): Initial energy level (0.0 to 1.0).
            initial_boredom (float): Initial boredom level (0.0 to 1.0).
        """
        self.energy = self._clamp(initial_energy)
        self.boredom = self._clamp(initial_boredom)

    def update(self, activity_level: float) -> None:
        """
        Updates the internal drives based on the current activity level.
        High activity decreases energy and decreases boredom.
        Low activity (or inactivity) could theoretically recover energy and increase boredom.

        Args:
            activity_level (float): The level of activity to process (e.g., 0.0 to 1.0).
        """
        # Decrease energy based on activity. E.g., activity of 1.0 decreases energy by 0.05.
        energy_drain = activity_level * 0.05
        self.energy = self._clamp(self.energy - energy_drain)

        # Decrease boredom based on activity. High activity reduces boredom more.
        # But naturally boredom might increase over time (not modeled here unless activity is very low/negative).
        # We will simply decrease boredom when there is activity.
        boredom_relief = activity_level * 0.1
        self.boredom = self._clamp(self.boredom - boredom_relief)

        # Simple rest mechanic: if activity is very low, recover energy slightly
        if activity_level < 0.1:
            self.energy = self._clamp(self.energy + 0.02)
            self.boredom = self._clamp(self.boredom + 0.05)

    def get_drives_summary(self) -> str:
        """
        Returns a string summary of the current drive states.

        Returns:
            str: A formatted string summarizing energy and boredom.
        """
        return f"Drives - Energy: {self.energy:.2f}, Boredom: {self.boredom:.2f}"

    def get_state(self) -> Dict[str, float]:
        """
        Returns the drives state as a dictionary.

        Returns:
            Dict[str, float]: The current state of drives.
        """
        return {
            "energy": self.energy,
            "boredom": self.boredom
        }

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamps a value between a minimum and maximum."""
        return max(min_val, min(max_val, value))
