import datetime

class PinealGland:
    """
    Pineal Gland module.
    Responsible for maintaining circadian rhythms and providing time-of-day context,
    which can affect the agent's energy levels and responses.
    """

    def __init__(self) -> None:
        """Initializes the Pineal Gland."""
        pass

    def get_time_context(self) -> str:
        """
        Determines the current time of day based on the system clock.

        Returns:
            str: A string representing the time of day ("night", "morning", "afternoon", "evening").
        """
        hour = self._get_current_time().hour

        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def get_energy_modifier(self) -> float:
        """
        Calculates an energy modifier based on the time of day.
        Higher modifiers represent periods of higher natural energy.

        Returns:
            float: An energy modifier (e.g., 1.2 for morning, 0.8 for night).
        """
        context = self.get_time_context()
        if context == "morning":
            return 1.2
        elif context == "afternoon":
            return 1.0
        elif context == "evening":
            return 0.9
        else: # night
            return 0.7

    def _get_current_time(self) -> datetime.datetime:
        """
        Returns the current datetime.
        Extracted as a separate method to allow easier mocking during tests.

        Returns:
            datetime.datetime: Current local time.
        """
        return datetime.datetime.now()
