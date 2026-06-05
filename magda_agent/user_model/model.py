import json
import os
from typing import Dict, Any, Optional
from magda_agent.llm_client import LLMClient
import logging

class UserModel:
    """
    Builds and maintains a persistent user model containing preferences, communication style,
    expertise level, and recurring topics.
    Updates incrementally after interactions using an LLM.
    """
    def __init__(self, persist_dir: str = "./user_models", llm: Optional[LLMClient] = None):
        self.persist_dir = persist_dir
        self.llm = llm
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)

    def get_model(self, user_id: int) -> Dict[str, Any]:
        """Retrieve the current model for a user."""
        path = self._get_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading user model for {user_id}: {e}")

        # Default model
        return {
            "preferences": {},
            "communication_style": "default",
            "expertise_level": "unknown",
            "recurring_topics": []
        }

    def _get_path(self, user_id: int) -> str:
        return os.path.join(self.persist_dir, f"user_{user_id}.json")

    async def update_model(self, user_id: int, interaction_text: str):
        """Update the user model incrementally based on a recent interaction."""
        current_model = self.get_model(user_id)

        if not self.llm:
            logging.warning("No LLM client available to update user model.")
            return

        prompt = f"""
        Given the following new interaction and the current user model, provide an updated user model in JSON format.
        Keep the output strictly as JSON. Extract any new preferences, update the communication style or expertise level if apparent,
        and add to recurring topics if new themes emerge. Do not drop existing information unless it's explicitly contradicted.

        Current Model:
        {json.dumps(current_model, indent=2)}

        New Interaction:
        {interaction_text}

        Return a JSON object with these keys: preferences (dict), communication_style (string), expertise_level (string), recurring_topics (list of strings).
        """

        try:
            response = await self.llm.chat_completion([
                {"role": "system", "content": "You update user profiles based on interactions. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ], temperature=0.1)

            # Very basic extraction if the LLM wraps it in markdown code blocks
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            updated_model = json.loads(json_str.strip())

            # Basic validation
            for key in ["preferences", "communication_style", "expertise_level", "recurring_topics"]:
                if key not in updated_model:
                    updated_model[key] = current_model.get(key)

            with open(self._get_path(user_id), "w", encoding="utf-8") as f:
                json.dump(updated_model, f, indent=2, ensure_ascii=False)

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response for user model update: {e}")
        except Exception as e:
            logging.error(f"Error during user model update: {e}")
