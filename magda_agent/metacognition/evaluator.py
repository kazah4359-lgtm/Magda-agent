import json
import logging
from typing import Optional, Dict, Any

from magda_agent.llm_client import LLMClient
from magda_agent.memory.storage import MemorySystem
from magda_agent.emotions.engine import PADState

class Evaluator:
    """
    Metacognition: Self-Evaluation module.
    Evaluates the agent's responses based on usefulness, accuracy, completeness, and emotional adequacy.
    """
    def __init__(self, llm: LLMClient, memory: MemorySystem):
        self.llm = llm
        self.memory = memory
        self.last_evaluation: Optional[Dict[str, Any]] = None

    async def evaluate_response(self, user_input: str, agent_response: str) -> Optional[Dict[str, Any]]:
        """
        Evaluates the generated response using the LLM and stores the result in memory.
        """
        prompt = (
            "Evaluate the following response given by an AI to a user's input.\n"
            "Score the response from 1 to 10 on the following criteria:\n"
            "- usefulness\n"
            "- accuracy\n"
            "- completeness\n"
            "- emotional_adequacy\n\n"
            f"User input: {user_input}\n"
            f"AI Response: {agent_response}\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "usefulness": 8,\n'
            '  "accuracy": 9,\n'
            '  "completeness": 7,\n'
            '  "emotional_adequacy": 8,\n'
            '  "average_score": 8.0,\n'
            '  "feedback": "A short sentence explaining the score"\n'
            "}"
        )
        messages = [{"role": "system", "content": prompt}]
        try:
            evaluation_text = await self.llm.chat_completion(messages, temperature=0.1)
            # Remove any markdown formatting (e.g. ```json)
            if "```" in evaluation_text:
                evaluation_text = evaluation_text.split("```")[1]
                if evaluation_text.startswith("json"):
                    evaluation_text = evaluation_text[4:]

            evaluation = json.loads(evaluation_text.strip())

            # Store in memory
            content = f"Evaluation of response to '{user_input[:20]}...': Avg Score: {evaluation.get('average_score')} - {evaluation.get('feedback')}"
            self.memory.add_memory(
                content=content,
                importance=0.6,
                emotional_state=PADState(0.0, 0.0, 0.0), # Neutral PAD state for evaluation
                tags=["evaluation", "metacognition"]
            )

            self.last_evaluation = evaluation
            return evaluation
        except Exception as e:
            logging.error(f"Failed to evaluate response: {e}")
            return None

    def get_feedback_for_prompt(self) -> str:
        """
        Returns feedback to be included in the next system prompt if the previous evaluation was low.
        """
        if not self.last_evaluation:
            return ""

        avg_score = self.last_evaluation.get("average_score", 10.0)
        if avg_score < 7.0:
            feedback = self.last_evaluation.get("feedback", "Improve response quality.")
            return f"Note: Your previous response received a low evaluation score ({avg_score}/10). Feedback: {feedback}. Please improve your response quality, accuracy, and emotional adequacy."
        return ""
