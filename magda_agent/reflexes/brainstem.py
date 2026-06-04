from typing import Optional

class Brainstem:
    """
    Brainstem module (Ствол мозга - Автономные рефлексы).
    Handles immediate autonomous responses to emergency commands like "stop", "help", or "emergency",
    bypassing the conscious processing (LLM) for rapid reaction.
    """

    def process_reflex(self, text: str) -> Optional[str]:
        """
        Process the input text to see if it triggers an autonomous reflex.

        Args:
            text (str): The input text from the user.

        Returns:
            Optional[str]: A rapid response string if a reflex is triggered, else None.
        """
        if not text:
            return None

        lower_text = text.lower().strip()

        if lower_text in ["stop", "стоп"]:
            return "Emergency Stop triggered. Halting current processes."
        elif lower_text in ["help", "помощь", "emergency", "экстренно"]:
            return "Emergency Assistance reflex triggered. How can I help you immediately?"

        return None
