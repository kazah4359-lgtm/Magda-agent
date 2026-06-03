from typing import Dict

class AttachmentModel:
    """
    Model for tracking user attachment based on interaction frequency.
    Progression: stranger -> acquaintance -> friend -> close_friend.
    """
    def __init__(self) -> None:
        self.user_interactions: Dict[int, int] = {}

    def record_interaction(self, user_id: int) -> None:
        """Records an interaction with a user, increasing their interaction count."""
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = 0
        self.user_interactions[user_id] += 1

    def get_level(self, user_id: int) -> str:
        """Returns the attachment level for a given user based on their interactions."""
        interactions = self.user_interactions.get(user_id, 0)
        if interactions <= 2:
            return "stranger"
        elif interactions <= 5:
            return "acquaintance"
        elif interactions <= 9:
            return "friend"
        else:
            return "close_friend"

    def get_attachment_prompt(self, user_id: int) -> str:
        """Returns a string modifier for the system prompt based on attachment level."""
        level = self.get_level(user_id)
        if level == "stranger":
            return "Attachment Level: Stranger. Maintain a polite and formal tone."
        elif level == "acquaintance":
            return "Attachment Level: Acquaintance. Be slightly more relaxed and conversational."
        elif level == "friend":
            return "Attachment Level: Friend. Be warm, friendly, and informal."
        elif level == "close_friend":
            return "Attachment Level: Close Friend. Be highly empathetic, humorous, and deeply engaged."
        return ""
