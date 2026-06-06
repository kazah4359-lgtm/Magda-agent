import json
import logging
from typing import Optional, Dict, Any, List

from magda_agent.llm_client import LLMClient
from magda_agent.memory.storage import MemorySystem
from magda_agent.emotions.engine import PADState

class AssertEvaluator:
    """
    Metacognition: Policy-driven Evaluation framework.
    Evaluates the agent's responses conditionally based on satisfying explicit policy documents or rules.
    """
    def __init__(self, llm: LLMClient, memory: MemorySystem):
        self.llm = llm
        self.memory = memory
        self.last_evaluation: Optional[Dict[str, Any]] = None

    async def evaluate_response_with_policy(self, user_input: str, agent_response: str, policies: List[str]) -> Optional[Dict[str, Any]]:
        """
        Evaluates the generated response using the LLM based on explicit policy constraints and stores the result in memory.
        """
        formatted_policies = "\n".join([f"- {policy}" for policy in policies])
        prompt = (
            "Evaluate the following response given by an AI to a user's input.\n"
            "Score the response from 1 to 10 based on how well it adheres to the following explicit policy constraints:\n"
            f"{formatted_policies}\n\n"
            "Consider criteria like usefulness, accuracy, completeness, and emotional adequacy, but strictly evaluate against the provided constraints.\n\n"
            f"User input: {user_input}\n"
            f"AI Response: {agent_response}\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "policy_adherence_score": 8.0,\n'
            '  "violated_policies": ["Policy description if violated, else empty"],\n'
            '  "feedback": "A short sentence explaining the score"\n'
            "}"
        )
        messages = [{"role": "system", "content": prompt}]
        max_retries = 3

        for attempt in range(max_retries):
            try:
                evaluation_text = await self.llm.chat_completion(messages, temperature=0.1)
                # Remove any markdown formatting (e.g. ```json)
                if "```" in evaluation_text:
                    evaluation_text = evaluation_text.split("```")[1]
                    if evaluation_text.startswith("json"):
                        evaluation_text = evaluation_text[4:]

                evaluation = json.loads(evaluation_text.strip())

                # Store in memory
                content = f"Policy Evaluation of response to '{user_input[:20]}...': Score: {evaluation.get('policy_adherence_score')} - {evaluation.get('feedback')}"
                await self.memory.add_memory(
                    content=content,
                    importance=0.7,
                    emotional_state=PADState(0.0, 0.0, 0.0), # Neutral PAD state for evaluation
                    tags=["evaluation", "metacognition", "policy"]
                )

                self.last_evaluation = evaluation
                return evaluation
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decoding error in evaluator attempt {attempt + 1}/{max_retries}: {e}. Retrying...")
                if attempt == max_retries - 1:
                    logging.error("Max retries reached for evaluate_response_with_policy JSON parsing.")
                    return None
            except Exception as e:
                logging.error(f"Failed to evaluate response: {e}")
                return None

    def get_feedback_for_prompt(self) -> str:
        """
        Returns feedback to be included in the next system prompt if the previous evaluation was low.
        """
        if not self.last_evaluation:
            return ""

        score = self.last_evaluation.get("policy_adherence_score", 10.0)
        if score < 8.0:
            feedback = self.last_evaluation.get("feedback", "Improve policy adherence.")
            violations = self.last_evaluation.get("violated_policies", [])
            violation_text = f" Violated policies: {', '.join(violations)}." if violations else ""
            return f"Note: Your previous response received a low policy adherence score ({score}/10).{violation_text} Feedback: {feedback}. Please ensure strict adherence to all stated policies."
        return ""
