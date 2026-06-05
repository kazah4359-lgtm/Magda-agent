from typing import Dict, Optional

class AttachmentModel:
    """
    Model for tracking user attachment based on interaction frequency.
    Progression: stranger -> acquaintance -> friend -> close_friend.
    """
    def __init__(self) -> None:
        self.user_interactions: Dict[int, int] = {}

    def reset(self, user_id: Optional[int] = None) -> None:
        """Resets the interaction count for a specific user (maps None to anonymous)."""
        u_id = user_id if user_id is not None else -1
        if u_id in self.user_interactions:
            del self.user_interactions[u_id]

    def reset_all(self) -> None:
        """Clears interaction counts for all users completely."""
        self.user_interactions.clear()

    def record_interaction(self, user_id: Optional[int] = None) -> None:
        """Records an interaction with a user, increasing their interaction count."""
        u_id = user_id if user_id is not None else -1
        if u_id not in self.user_interactions:
            self.user_interactions[u_id] = 0
        self.user_interactions[u_id] += 1

    def get_level(self, user_id: Optional[int] = None) -> str:
        """Returns the attachment level for a given user based on their interactions."""
        u_id = user_id if user_id is not None else -1
        interactions = self.user_interactions.get(u_id, 0)
        if interactions <= 2:
            return "stranger"
        elif interactions <= 5:
            return "acquaintance"
        elif interactions <= 9:
            return "friend"
        else:
            return "close_friend"

    def get_attachment_prompt(self, user_id: Optional[int] = None) -> str:
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
