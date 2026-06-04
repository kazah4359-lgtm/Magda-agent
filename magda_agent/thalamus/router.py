import logging

class Thalamus:
    """
    Sensory filter module (Thalamus).
    Acts as a gateway for incoming messages before they reach consciousness.
    It filters out noise, empty messages, or extremely short nonsensical inputs.
    """
    def __init__(self):
        pass

    def filter_input(self, text: str) -> bool:
        """
        Filters the input text.
        Returns True if the message should be processed, False if it should be ignored.

        Args:
            text (str): The incoming user message.
        """
        if not text:
            logging.debug("Thalamus dropped empty message.")
            return False

        clean_text = text.strip()

        # Filter out empty or whitespace-only strings
        if not clean_text:
            logging.debug("Thalamus dropped whitespace-only message.")
            return False

        # Optional: Filter out single character noise, unless it's a valid query like '?'
        if len(clean_text) == 1 and not clean_text.isalpha() and clean_text not in ["?", "!", "."]:
            logging.debug("Thalamus dropped single non-alphabetic noise character.")
            return False

        return True
