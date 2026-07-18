import logging
import re
from typing import Dict, Any, Optional
from magda_agent.llm_client import LLMClient

class OnlineRLFeedbackLoop:
    """
    Component that captures user feedback signals (e.g. positive/negative replies)
    and maps them to Q-value updates for skills.
    Inspired by OpenClaw-RL.
    """

    def __init__(self, db_path: str = ":memory:", llm_client: Optional[LLMClient] = None) -> None:
        """
        Initializes the OnlineRLFeedbackLoop.

        Args:
            db_path (str): The path to the SQLite database. Defaults to in-memory.
            llm_client (Optional[LLMClient]): Optional LLMClient to extract rewards using LLM.
        """
        self.q_table: Dict[str, float] = {}
        self.llm_client = llm_client
        logging.info("Initialized OnlineRLFeedbackLoop with LLM Support")

    def process_feedback(self, skill_id: str, user_reply: str) -> None:
        """
        Processes user feedback and updates the Q-value for a given skill.

        Args:
            skill_id (str): The identifier of the skill used.
            user_reply (str): The text of the user's reply.
        """
        reward = self._map_reply_to_reward(user_reply)

        current_q = self.q_table.get(skill_id, 0.0)

        # Simple Q-learning update rule with alpha=0.1, gamma=0 (immediate reward only)
        alpha = 0.1
        new_q = current_q + alpha * (reward - current_q)

        self.q_table[skill_id] = new_q
        logging.info(f"Updated Q-value for {skill_id}: {new_q:.2f} (Reward: {reward})")

    async def process_feedback_async(
        self,
        skill_id: str,
        user_reply: str,
        tool_output: Optional[str] = None
    ) -> None:
        """
        Asynchronously processes user feedback, utilizing LLM to evaluate the reward
        from next-state signals (user reply and optional tool output).

        Args:
            skill_id (str): The identifier of the skill used.
            user_reply (str): The text of the user's reply.
            tool_output (Optional[str]): Optional output from the tool execution.
        """
        reward = await self.extract_reward_llm(user_reply, tool_output)

        current_q = self.get_q_value(skill_id)
        alpha = 0.1
        new_q = current_q + alpha * (reward - current_q)

        self.q_table[skill_id] = new_q
        logging.info(f"Updated Q-value for {skill_id} asynchronously: {new_q:.2f} (Reward: {reward})")

    async def extract_reward_llm(self, user_reply: str, tool_output: Optional[str] = None) -> float:
        """
        Uses LLM to evaluate conversational next-state signals (user reply and optional tool output)
        to extract a reward between -1.0 and 1.0 representing user satisfaction / success.
        If no LLMClient is available, falls back to the heuristic mapping.

        Args:
            user_reply (str): The user reply text.
            tool_output (Optional[str]): The optional tool output text.

        Returns:
            float: A reward score between -1.0 and 1.0.
        """
        if not self.llm_client:
            return self._map_reply_to_reward(user_reply)

        prompt = (
            "Analyze the following user response and optional tool output to determine user satisfaction.\n"
            f"User Reply: {user_reply}\n"
        )
        if tool_output:
            prompt += f"Tool Output: {tool_output}\n"
        prompt += (
            "Based on the interaction, output a single floating-point number representing the reward "
            "from -1.0 (extremely dissatisfied, incorrect result, angry user) to 1.0 (very satisfied, "
            "thankful user, successful tool execution).\n"
            "If neutral, return 0.0.\n"
            "Return ONLY the floating point number as your response, nothing else."
        )

        try:
            response = await self.llm_client.generate(prompt)
            cleaned = response.strip()
            match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
            if match:
                reward = float(match.group())
                return max(-1.0, min(1.0, reward))
            return 0.0
        except Exception as e:
            logging.error(f"Error extracting reward via LLM: {e}")
            return self._map_reply_to_reward(user_reply)

    def _map_reply_to_reward(self, reply: str) -> float:
        """
        Maps a user reply text to a numerical reward score.

        Args:
            reply (str): The user reply.

        Returns:
            float: The reward score.
        """
        reply_lower = reply.lower()

        # Check negative first, but make sure it's not matching 'know' incorrectly due to 'no'
        # Tokenize by word
        words = reply_lower.replace("'", "").replace(".", "").replace(",", "").replace("!", "").split()

        if any(w in ["good", "great", "thanks", "awesome", "yes"] for w in words):
            return 1.0
        elif any(w in ["bad", "terrible", "wrong", "no"] for w in words):
            return -1.0
        else:
            return 0.0

    def get_q_value(self, skill_id: str) -> float:
        """
        Retrieves the current Q-value for a given skill.

        Args:
            skill_id (str): The identifier of the skill.

        Returns:
            float: The current Q-value.
        """
        return self.q_table.get(skill_id, 0.0)
