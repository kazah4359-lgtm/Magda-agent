import logging
from typing import Optional, Dict

class HabitTracker:
    """
    Cerebellum: Learning from habits.
    Analyzes patterns to find which skills are frequently used for typical requests
    and receive high evaluation scores. Forms 'habits' - preferred strategies.
    """

    def __init__(self):
        # Maps input_text to a dictionary mapping skill_used to success counts
        # e.g., {"what is the time": {"get_time": 5, "search_web": 1}}
        self.habits: Dict[str, Dict[str, int]] = {}

    def record_usage(self, input_text: str, skill_used: str, evaluation_score: float) -> None:
        """
        Records the successful usage of a skill for a given input.

        Args:
            input_text (str): The user's input.
            skill_used (str): The name of the skill that was used.
            evaluation_score (float): The evaluation score of the response.
        """
        # We only form habits from successful responses
        if evaluation_score >= 8.0:
            if input_text not in self.habits:
                self.habits[input_text] = {}

            if skill_used not in self.habits[input_text]:
                self.habits[input_text][skill_used] = 0

            self.habits[input_text][skill_used] += 1
            logging.info(f"Habit reinforced: For input '{input_text[:20]}...', skill '{skill_used}' count is now {self.habits[input_text][skill_used]}")

    def suggest_strategy(self, input_text: str) -> Optional[str]:
        """
        Suggests a preferred skill based on past high-scoring experiences for similar inputs.

        Args:
            input_text (str): The user's input to find a strategy for.

        Returns:
            Optional[str]: The name of the suggested skill, or None if no strong habit exists.
        """
        if input_text not in self.habits:
            return None

        skill_counts = self.habits[input_text]
        if not skill_counts:
            return None

        # Find the skill with the highest success count
        best_skill = max(skill_counts, key=skill_counts.get)
        max_count = skill_counts[best_skill]

        # Require a threshold of success before suggesting (e.g., at least 2 successful uses)
        if max_count >= 2:
            logging.info(f"Habit matched: Suggesting skill '{best_skill}' for input '{input_text[:20]}...'")
            return best_skill

        return None
