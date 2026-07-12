from typing import Dict, Any, List, Optional
import logging
import json

from magda_agent.llm_client import LLMClient
from magda_agent.agents.sub_agent import SubAgent


class EvaluatorAgentChunkingV3:
    """
    Evaluator Agent capable of handling large context windows using a map-reduce pattern.

    This agent splits the large output into chunks, evaluates each chunk individually (map),
    and then aggregates the results (reduce).
    """

    def __init__(self, llm: LLMClient, chunk_size: int = 10000) -> None:
        """
        Initializes the EvaluatorAgentChunkingV3.

        Args:
            llm (LLMClient): The LLM client to use for evaluations.
            chunk_size (int): The maximum size of each chunk in characters. Defaults to 10000.
        """
        self.llm = llm
        self.chunk_size = chunk_size
        self.sub_agent = SubAgent(
            llm=self.llm,
            system_prompt="You are a strict QA engineer evaluating generated output.",
            use_isolation=False
        )

    def _chunk_text(self, text: str) -> List[str]:
        """
        Splits a large text into smaller chunks based on the chunk_size.

        Args:
            text (str): The large text to be chunked.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            return [""]
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    async def evaluate_output(self,
                               generator_output: str,
                               user_request: str,
                               task_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates a large output by chunking it and applying a map-reduce evaluation pattern.

        Args:
            generator_output (str): The large text or code output to evaluate.
            user_request (str): The initial user request or instruction.
            task_context (Optional[Dict[str, Any]]): Additional task context or metadata.

        Returns:
            Dict[str, Any]: The aggregated evaluation result containing 'score', 'approved', and 'feedback'.
        """
        logging.info("Starting map-reduce evaluation for large context window...")

        chunks = self._chunk_text(generator_output)

        # Map Phase: Evaluate each chunk
        chunk_evals: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            result = await self._evaluate_chunk(chunk, user_request, task_context, i, len(chunks))
            chunk_evals.append(result)

        # Reduce Phase: Aggregate scores
        return await self._reduce_evaluations(chunk_evals, user_request, task_context)

    async def _evaluate_chunk(self,
                               chunk: str,
                               request: str,
                               context: Optional[Dict[str, Any]],
                               index: int,
                               total: int) -> Dict[str, Any]:
        """
        Evaluates a single chunk of the generated output.

        Args:
            chunk (str): The chunk of text to evaluate.
            request (str): The original user request.
            context (Optional[Dict[str, Any]]): Additional task context.
            index (int): The current chunk index.
            total (int): The total number of chunks.

        Returns:
            Dict[str, Any]: The evaluation result for this chunk.
        """
        task = (
            f"Review part {index+1} of {total} of the output against the user request.\n\n"
            "Provide evaluation ONLY as JSON:\n"
            '{"score": 1-10, "approved": bool, "feedback": "Detailed reasoning"}'
        )

        context_str = (
            f"User Request: {request}\n"
            f"--- CHUNK OUTPUT ---\n{chunk}\n------------------------\n"
        )

        try:
            response_text = await self.sub_agent.execute(task=task, context=context_str, temperature=0.1)
            # Basic JSON extraction
            if "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 3:
                    response_text = parts[1]
                    if response_text.strip().startswith("json"):
                        response_text = response_text.strip()[4:]

            return json.loads(response_text.strip())
        except Exception as e:
            logging.error(f"Failed to evaluate chunk {index}: {e}")
            return {"score": 0.0, "approved": False, "feedback": f"Evaluation error: {e}"}

    async def _reduce_evaluations(self,
                                   chunk_evals: List[Dict[str, Any]],
                                   request: str,
                                   context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates the evaluations from all chunks into a final decision.

        Args:
            chunk_evals (List[Dict[str, Any]]): The list of chunk evaluations.
            request (str): The original user request.
            context (Optional[Dict[str, Any]]): Additional task context.

        Returns:
            Dict[str, Any]: The final aggregated evaluation.
        """
        if not chunk_evals:
            return {"score": 0.0, "approved": False, "feedback": "No evaluations completed."}

        total_score = sum(float(ev.get("score", 0.0)) for ev in chunk_evals)
        avg_score = total_score / len(chunk_evals)

        # Approve only if all chunks meet the approval criteria
        all_approved = all(bool(ev.get("approved", False)) for ev in chunk_evals)

        combined_feedback = "\n\n".join(
            f"[Chunk {i+1} Feedback]: {ev.get('feedback', '')}"
            for i, ev in enumerate(chunk_evals)
        )

        return {
            "score": round(avg_score, 2),
            "approved": all_approved,
            "feedback": f"Aggregated Map-Reduce Feedback:\n\n{combined_feedback}",
            "metadata": {
                "chunks_processed": len(chunk_evals)
            }
        }
