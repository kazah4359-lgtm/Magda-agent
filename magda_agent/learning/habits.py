import logging
import uuid
import chromadb
from typing import Optional, Dict
from collections import Counter

class HabitTracker:
    """
    Cerebellum: Learning from habits.
    Analyzes patterns to find which skills are frequently used for typical requests
    and receive high evaluation scores. Forms 'habits' - preferred strategies
    using semantic similarity search.
    """

    def __init__(self, persist_directory: str = "./habits_db") -> None:
        """
        Initializes the HabitTracker with a ChromaDB client.

        Args:
            persist_directory (str): The directory to persist ChromaDB data.
                                     Use ":memory:" for an ephemeral client.
        """
        if persist_directory == ":memory:":
            self.client = chromadb.EphemeralClient()
            logging.info("Initialized HabitTracker with EphemeralClient")
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logging.info(f"Initialized HabitTracker with persistent directory: {persist_directory}")

        # We store successful habit mappings as documents in ChromaDB
        self.collection = self.client.get_or_create_collection(name="habits")

    def record_usage(self, input_text: str, skill_used: str, evaluation_score: float, user_id: int = None) -> None:
        """
        Records the successful usage of a skill for a given input.

        Args:
            input_text (str): The user's input.
            skill_used (str): The name of the skill that was used.
            evaluation_score (float): The evaluation score of the response.
            user_id (int, optional): The ID of the user.
        """
        # We only form habits from successful responses
        if evaluation_score >= 8.0:
            try:
                habit_id = str(uuid.uuid4())
                metadata = {"skill_used": skill_used}
                if user_id is not None:
                    metadata["user_id"] = user_id
                self.collection.add(
                    documents=[input_text],
                    metadatas=[metadata],
                    ids=[habit_id]
                )
                logging.info(f"Habit reinforced: Stored success for skill '{skill_used}' with input '{input_text[:20]}...'")
            except Exception as e:
                logging.error(f"Failed to record habit: {e}")

    def suggest_strategy(self, input_text: str, user_id: int = None) -> Optional[str]:
        """
        Suggests a preferred skill based on past high-scoring experiences for similar inputs.

        Args:
            input_text (str): The user's input to find a strategy for.
            user_id (int, optional): The ID of the user.

        Returns:
            Optional[str]: The name of the suggested skill, or None if no strong habit exists.
        """
        try:
            # Need to catch exception if collection is empty
            if self.collection.count() == 0:
                return None

            query_kwargs = {
                "query_texts": [input_text],
                "n_results": min(5, self.collection.count())
            }
            if user_id is not None:
                query_kwargs["where"] = {"user_id": user_id}

            results = self.collection.query(**query_kwargs)

            if not results or not results.get("distances") or not results["distances"][0]:
                return None

            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            valid_skills = []
            # We set a threshold for distance (e.g., < 1.0 is reasonably close in Chroma defaults)
            # A distance of 0.0 is an exact match. Let's use 1.0 as a threshold for semantic similarity.
            distance_threshold = 1.0

            for dist, meta in zip(distances, metadatas):
                if dist < distance_threshold and meta and "skill_used" in meta:
                    valid_skills.append(meta["skill_used"])

            if not valid_skills:
                return None

            skill_counts = Counter(valid_skills)
            best_skill, max_count = skill_counts.most_common(1)[0]

            # Require a threshold of success before suggesting (e.g., at least 2 successful uses)
            if max_count >= 2:
                logging.info(f"Habit matched: Suggesting skill '{best_skill}' for input '{input_text[:20]}...'")
                return best_skill

            return None
        except Exception as e:
            logging.error(f"Failed to suggest strategy: {e}")
            return None
