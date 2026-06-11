import json
import os
import time
import logging
from typing import Dict, Any, List, Optional
from magda_agent.llm_client import LLMClient

class HermesPersistentMemory:
    """
    Persistent user profile memory inspired by Hermes Agent.
    Accumulates traits, facts, and stats about a user over time.
    """
    def __init__(self, persist_dir: str = "./user_profiles", llm: Optional[LLMClient] = None):
        self.persist_dir = persist_dir
        self.llm = llm
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)

    def get_profile(self, user_id: int) -> Dict[str, Any]:
        """Retrieve the persistent profile for a specific user."""
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
            "stats": {
                "interaction_count": 0,
                "first_seen": time.time(),
                "last_seen": time.time()
            },
            "version": 1
        }

    def _get_path(self, user_id: int) -> str:
        return os.path.join(self.persist_dir, f"hermes_{user_id}.json")

    def save_profile(self, user_id: int, profile: Dict[str, Any]):
        """Save the user profile to disk."""
        path = self._get_path(user_id)
        try:
            profile["stats"]["last_seen"] = time.time()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving Hermes profile for {user_id}: {e}")

    async def update_from_interaction(self, user_id: int, interaction_text: str):
        """
        Update the persistent profile based on a new interaction.
        Uses an LLM to extract stable traits and facts.
        """
        profile = self.get_profile(user_id)
        profile["stats"]["interaction_count"] += 1

        if not self.llm:
            logging.warning("No LLM client available for Hermes profile update.")
            self.save_profile(user_id, profile)
            return

        prompt = f"""
        Analyze the following interaction and update the user profile.
        Identify any new stable traits (behavioral patterns, preferences) or facts about the user.
        Return ONLY a JSON object with new traits and facts to be added.
        Do not remove existing information unless it's explicitly contradicted.

        Current Profile:
        {json.dumps(profile, indent=2)}

        New Interaction:
        {interaction_text}

        Output format: {{"traits": ["new_trait1", ...], "facts": ["new_fact1", ...]}}
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
            new_traits = updates.get("traits", [])
            for trait in new_traits:
                if trait not in profile["traits"]:
                    profile["traits"].append(trait)

            new_facts = updates.get("facts", [])
            for fact in new_facts:
                if fact not in profile["facts"]:
                    profile["facts"].append(fact)

            self.save_profile(user_id, profile)
            logging.info(f"Updated Hermes profile for user {user_id}")
        except Exception as e:
            logging.error(f"Error updating Hermes profile: {e}")
            self.save_profile(user_id, profile)
