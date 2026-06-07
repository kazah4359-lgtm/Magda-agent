import logging
from typing import Optional
from magda_agent.emotions.mirror_neurons import MirrorNeurons
from magda_agent.user_model.model import UserModel

class FeedbackLoop:
    """
    Feedback loop for online reinforcement learning from continuous dialogue.
    Adjusts style padding and prompt weights based on implicit and explicit feedback.
    """
    def __init__(self, mirror_neurons: MirrorNeurons, user_model: UserModel) -> None:
        self.mirror_neurons = mirror_neurons
        self.user_model = user_model

    async def process_feedback(self, user_reply: str, user_id: int) -> None:
        """
        Analyzes the user's reply, updates communication style and weights in the user model.

        Args:
            user_reply (str): The text of the user's reply.
            user_id (int): The user's ID.
        """
        if not user_reply:
            return

        p_shift, a_shift, d_shift = self.mirror_neurons.empathize(user_reply)
        model_data = self.user_model.get_model(user_id)

        if p_shift > 0.0:
            logging.info(f"FeedbackLoop: Positive feedback received (p_shift={p_shift:.2f}).")
            if "(friendly)" not in model_data.get("communication_style", ""):
                current_style = model_data.get("communication_style", "default")
                model_data["communication_style"] = f"{current_style} (friendly)".strip()
        elif p_shift < 0.0:
            logging.info(f"FeedbackLoop: Negative feedback received (p_shift={p_shift:.2f}).")
            if "(cautious)" not in model_data.get("communication_style", ""):
                current_style = model_data.get("communication_style", "default")
                model_data["communication_style"] = f"{current_style} (cautious)".strip()

        if "preferences" not in model_data:
            model_data["preferences"] = {}
        model_data["preferences"]["last_p_shift"] = p_shift

        self.user_model.save_model(user_id, model_data)
        logging.info(f"FeedbackLoop: Updated user model for user {user_id}")
