from typing import Dict, Any, Optional
import logging
import json

from magda_agent.llm_client import LLMClient
from magda_agent.agents.sub_agent import SubAgent


class EvaluatorAgentV2(SubAgent):
    """
    Claude-inspired specialized sub-agent for evaluating output from Generator sub-agents.
    Inherits from SubAgent for isolated execution capability.
    """
    def __init__(self, llm: LLMClient):
        super().__init__(
            llm=llm,
            system_prompt=(
                "You are an expert code and output Evaluator sub-agent. "
                "Critique the provided output against the user request and provide actionable feedback. "
                "You MUST output valid JSON only."
            ),
            use_isolation=False
        )

    async def evaluate_generator_output(self, generator_output: str, user_request: str) -> Dict[str, Any]:
        """
        Evaluates the output produced by the Generator Agent and provides critique.

        Args:
            generator_output (str): The text or code output generated.
            user_request (str): The initial user request or context.

        Returns:
            Dict[str, Any]: Parsed JSON representing the critique and feedback containing keys:
                - score (int): 1-10 evaluation score.
                - approved (bool): Whether the output is acceptable.
                - feedback (str): Detailed, actionable critique.
        """
        logging.info("Starting EvaluatorAgentV2 evaluation...")

        task = (
            "Evaluate the generator's output against the user's original request.\n\n"
            "Return a JSON object with the following schema:\n"
            '{"score": <int 1-10>, "approved": <bool>, "feedback": "<detailed critique and actionable feedback>"}'
        )

        context_str = (
            f"User Request:\n{user_request}\n"
            "------------------------\n"
            f"Generator Output:\n{generator_output}\n"
            "------------------------\n"
        )

        try:
            response_text = await self.execute(task=task, context=context_str, temperature=0.2)

            # Simple markdown cleaning if the LLM wraps it in markdown blocks
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.lower().startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text.strip())

            # Validate expected keys are present
            if not all(k in result for k in ("score", "approved", "feedback")):
                raise ValueError("Missing required keys in LLM JSON output")

            return result

        except Exception as e:
            logging.error(f"EvaluatorAgentV2 review failed: {e}")
            return {
                "score": 0,
                "approved": False,
                "feedback": f"Evaluation error: {str(e)}"
            }
