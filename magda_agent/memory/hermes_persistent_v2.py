import json
import os
import time
import logging
from typing import Dict, Any, Optional
from magda_agent.llm_client import LLMClient

class HermesPersistentMemoryV2:
    """
    Persistent user profile memory V2 inspired by Hermes Agent.
    Accumulates traits, facts, stats, interests, communication_style, and goals about a user over time.
    """
    def __init__(self, persist_dir: str = "./user_profiles_v2", llm: Optional[LLMClient] = None) -> None:
        """
        Initializes the HermesPersistentMemoryV2.

        Args:
            persist_dir (str): The directory to store user profiles.
            llm (Optional[LLMClient]): The LLM client to use for profile updates.
        """
        self.persist_dir = persist_dir
        self.llm = llm
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)

    def get_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Retrieve the persistent profile for a specific user.

        Args:
            user_id (int): The ID of the user.

        Returns:
            Dict[str, Any]: The user profile.
        """
        path = self._get_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading Hermes profile for {user_id}: {e}")

        # Initial profile
        return {
            "user_id": user_id,
            "traits": [], # E.g. "prefers concise answers", "highly technical"
            "facts": [],  # E.g. "Lives in Berlin", "Uses Python"
            "interests": [], # E.g. "AI agents", "distributed systems"
            "communication_style": [], # E.g. "formal", "uses emojis"
            "goals": [], # E.g. "learn Rust", "build a startup"
            "stats": {
                "interaction_count": 0,
                "first_seen": time.time(),
                "last_seen": time.time()
            },
            "version": 2
        }

    def _get_path(self, user_id: int) -> str:
        """
        Get the file path for a user's profile.

        Args:
            user_id (int): The ID of the user.

        Returns:
            str: The file path.
        """
        return os.path.join(self.persist_dir, f"hermes_v2_{user_id}.json")

    def save_profile(self, user_id: int, profile: Dict[str, Any]) -> None:
        """
        Save the user profile to disk.

        Args:
            user_id (int): The ID of the user.
            profile (Dict[str, Any]): The profile to save.
        """
        path = self._get_path(user_id)
        try:
            profile["stats"]["last_seen"] = time.time()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving Hermes profile for {user_id}: {e}")

    async def update_from_interaction(self, user_id: int, interaction_text: str) -> None:
        """
        Update the persistent profile based on a new interaction.
        Uses an LLM to extract stable traits, facts, interests, communication_style, and goals.

        Args:
            user_id (int): The ID of the user.
            interaction_text (str): The text of the interaction.
        """
        profile = self.get_profile(user_id)
        profile["stats"]["interaction_count"] += 1

        if not self.llm:
            logging.warning("No LLM client available for Hermes profile update.")
            self.save_profile(user_id, profile)
            return

        prompt = f"""
        Analyze the following interaction and update the user profile.
        Identify any new stable traits (behavioral patterns, preferences), facts, interests, communication style, or goals about the user.
        Return ONLY a JSON object with new traits, facts, interests, communication_style, and goals to be added.
        Do not remove existing information unless it's explicitly contradicted.

        Current Profile:
        {json.dumps(profile, indent=2)}

        New Interaction:
        {interaction_text}

        Output format: {{"traits": ["new_trait1", ...], "facts": ["new_fact1", ...], "interests": ["new_interest1", ...], "communication_style": ["new_style1", ...], "goals": ["new_goal1", ...]}}
        """

        try:
            response = await self.llm.chat_completion([
                {"role": "system", "content": "You maintain user profiles. Return strictly valid JSON."},
                {"role": "user", "content": prompt}
            ], temperature=0.1)

            json_str = response.strip()
            # Basic JSON extraction from markdown
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            updates = json.loads(json_str.strip())

            # Merge updates ensuring uniqueness
            keys_to_update = ["traits", "facts", "interests", "communication_style", "goals"]
            for key in keys_to_update:
                new_items = updates.get(key, [])
                for item in new_items:
                    if item not in profile[key]:
                        profile[key].append(item)

            self.save_profile(user_id, profile)
            logging.info(f"Updated Hermes V2 profile for user {user_id}")
        except Exception as e:
            logging.error(f"Error updating Hermes V2 profile: {e}")
            self.save_profile(user_id, profile)
