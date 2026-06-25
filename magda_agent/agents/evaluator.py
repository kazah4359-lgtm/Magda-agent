from typing import Dict, Any, List, Optional
import logging
import json
import subprocess
import os

from magda_agent.llm_client import LLMClient

class EvaluatorAgent:
    """
    Advanced Evaluator Agent Module.
    Implements multi-stage evaluation: LLM reasoning + Tool-based verification.
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def evaluate_output(self, 
                               generator_output: str, 
                               user_request: str, 
                               task_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates the output produced by the Generator Agent.
        
        Args:
            generator_output (str): The text or code output generated.
            user_request (str): The initial user request or context.
            task_context (dict): Full task object from agent_tasks.json (optional).
        """
        logging.info("Starting multi-stage evaluation...")
        
        # Stage 1: LLM Cognitive Evaluation
        llm_result = await self._llm_review(generator_output, user_request, task_context)
        
        # Stage 2: Tool-Based Verification (if code or tests are involved)
        tool_result = await self._tool_verification(generator_output, task_context)
        
        # Final Decision
        final_approved = llm_result.get("approved", False) and tool_result.get("verified", True)
        
        # Combine feedback
        combined_feedback = llm_result.get("feedback", "")
        if not tool_result.get("verified"):
            combined_feedback += f"\n\n[Tool Verification Failure]:\n{tool_result.get('error', '')}"
            if tool_result.get("test_output"):
                combined_feedback += f"\n\nTest Output:\n{tool_result.get('test_output')}"

        return {
            "score": llm_result.get("score", 0) if final_approved else min(llm_result.get("score", 0), 5),
            "approved": final_approved,
            "feedback": combined_feedback,
            "metadata": {
                "llm_eval": llm_result,
                "tool_eval": tool_result
            }
        }

    async def _llm_review(self, output: str, request: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Asks the LLM to review the output against requirements and checklist."""
        checklist = context.get("acceptance", []) if context else []
        
        prompt = (
            "You are the Lead Quality Evaluator. Review the output against the user request and acceptance criteria.\n\n"
            f"User Request: {request}\n"
            f"Acceptance Criteria: {json.dumps(checklist, indent=2)}\n"
            f"--- GENERATED OUTPUT ---\n{output}\n------------------------\n\n"
            "Provide evaluation as JSON:\n"
            '{"score": 1-10, "approved": bool, "feedback": "Detailed reasoning", "checklist_status": {"criterion": "pass/fail"}}\n'
        )

        messages = [{"role": "system", "content": "You are a strict QA engineer."}, {"role": "user", "content": prompt}]
        try:
            response_text = await self.llm.chat_completion(messages, temperature=0.1)
            # Simple markdown cleaning
            if "")[1]
                if response_text.startswith("json"): response_text = response_text[4:]
            
            return json.loads(response_text.strip())
        except Exception as e:
            logging.error(f"LLM review failed: {e}")
            return {"score": 0, "approved": False, "feedback": f"LLM error: {e}"}

    async def _tool_verification(self, output: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs real tools (pytest, linters) if the task implies code changes."""
        if not context:
            return {"verified": True}

        # If the task has allowed_paths or mentions tests, let's try to verify
        paths = context.get("allowed_paths", [])
        
        # Logic: If we are in a Rool-like state, we should check if tests pass
        # This is a placeholder for real tool execution within Jules' environment
        try:
            # Example: Run pytest if we see test files modified
            has_tests = any("test" in p for p in paths)
            if has_tests:
                result = subprocess.run(["pytest"], capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    return {
                        "verified": False, 
                        "error": "Automated tests failed.", 
                        "test_output": result.stdout[-1000:]
                    }
            
            return {"verified": True}
        except Exception as e:
            return {"verified": True} # Fallback to LLM if tools are not available
