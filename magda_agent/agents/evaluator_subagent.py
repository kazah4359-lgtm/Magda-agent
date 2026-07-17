import json
import logging
from typing import Optional, Dict, Any, List

from magda_agent.llm_client import LLMClient
from magda_agent.memory.storage import MemorySystem
from magda_agent.emotions.engine import PADState
from magda_agent.agents.sub_agent import SubAgent
from magda_agent.isolation.git_worktree_multi import GitWorktreeMultiManager
from magda_agent.planning.dependency_graph import DependencyGraph

class EvaluatorSubagent:
    """
    Evaluator implemented as a SubAgent with isolation.
    Evaluates the agent's responses based on usefulness, accuracy, completeness, and emotional adequacy.
    Inspired by Claude Agent SDK Subagent evaluation pattern.
    """
    def __init__(self, llm: LLMClient, memory: MemorySystem, worktree_manager: Optional[GitWorktreeMultiManager] = None):
        """
        Initializes the EvaluatorSubagent.

        Args:
            llm: The Language Model client.
            memory: The memory system to store evaluations.
            worktree_manager: Optional GitWorktreeMultiManager for multi-agent isolation.
        """
        self.llm = llm
        self.memory = memory
        self.worktree_manager = worktree_manager or GitWorktreeMultiManager()
        self.last_evaluation: Optional[Dict[str, Any]] = None
        self.sub_agent = SubAgent(
            llm=llm,
            system_prompt="You are an isolated SubAgent responsible for strictly evaluating responses and plans.",
            use_isolation=True
        )

    async def evaluate_response(self, user_input: str, agent_response: str, policies: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Evaluates the generated response using the LLM via SubAgent execute and stores the result in memory.

        Args:
            user_input: The user's original input message.
            agent_response: The agent's generated response to evaluate.
            policies: Optional list of policies to evaluate against.

        Returns:
            The evaluation dictionary or None if evaluation fails.
        """
        policy_str = ""
        if policies:
            policy_str = "\nEvaluate against these specific policies:\n" + "\n".join([f"- {p}" for p in policies])

        task = (
            "Evaluate the response given by an AI to a user's input.\n"
            "Score the response from 1 to 10 on the following criteria:\n"
            "- usefulness\n"
            "- accuracy\n"
            "- completeness\n"
            "- emotional_adequacy\n"
            f"{policy_str}\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "usefulness": 8,\n'
            '  "accuracy": 9,\n'
            '  "completeness": 7,\n'
            '  "emotional_adequacy": 8,\n'
            '  "policy_evaluations": {"policy_name": {"score": 8, "feedback": "reasoning"}},\n'
            '  "average_score": 8.0,\n'
            '  "feedback": "A short sentence explaining the overall score"\n'
            "}"
        )

        context = (
            f"User input: {user_input}\n"
            f"AI Response: {agent_response}"
        )

        max_retries = 3

        for attempt in range(max_retries):
            try:
                evaluation_text = await self.sub_agent.execute(task=task, context=context, temperature=0.1)

                # Remove any markdown formatting (e.g. ```json)
                if "```" in evaluation_text:
                    evaluation_text = evaluation_text.split("```")[1]
                    if evaluation_text.startswith("json"):
                        evaluation_text = evaluation_text[4:]

                evaluation = json.loads(evaluation_text.strip())

                # Store in memory
                content = f"Evaluation of response to '{user_input[:20]}...': Avg Score: {evaluation.get('average_score')} - {evaluation.get('feedback')}"
                await self.memory.add_memory(
                    content=content,
                    importance=0.6,
                    emotional_state=PADState(0.0, 0.0, 0.0), # Neutral PAD state for evaluation
                    tags=["evaluation", "metacognition", "subagent"]
                )

                self.last_evaluation = evaluation
                return evaluation
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decoding error in evaluator attempt {attempt + 1}/{max_retries}: {e}. Retrying...")
                if attempt == max_retries - 1:
                    logging.error("Max retries reached for evaluate_response JSON parsing.")
                    return None
            except Exception as e:
                logging.error(f"Failed to evaluate response: {e}")
                return None

    def get_feedback_for_prompt(self) -> str:
        """
        Returns feedback to be included in the next system prompt if the previous evaluation was low.

        Returns:
            Feedback string or empty string if no evaluation or high score.
        """
        if not self.last_evaluation:
            return ""

        avg_score = self.last_evaluation.get("average_score", 10.0)
        if avg_score < 7.0:
            feedback = self.last_evaluation.get("feedback", "Improve response quality.")
            return f"Note: Your previous response received a low evaluation score ({avg_score}/10). Feedback: {feedback}. Please improve your response quality, accuracy, and emotional adequacy."
        return ""

    async def evaluate_planner_graph(self, plan: Dict[str, Any]) -> List[Any]:
        """
        Spawns a subagent in an isolated git worktree via GitWorktreeMultiManager
        to review a Planner dependency graph.

        Args:
            plan: The plan containing steps and dependencies.

        Returns:
            A list of evaluation results.
        """
        # Validate dependency graph
        steps = plan.get("steps", [])
        try:
            DependencyGraph.topological_sort(steps)
            structural_status = "Valid topological sort."
        except ValueError as e:
            structural_status = f"Cycle detected: {e}"

        context = f"Plan structural validation: {structural_status}\nSteps:\n" + json.dumps(steps, indent=2)
        task = (
            "Review the provided AI Agent dependency graph for logical soundness and parallelization opportunities. "
            "Evaluate if steps with missing dependencies can lead to failure. "
            "Output your findings as a JSON object."
        )

        # Create a SubAgent that doesn't use its own worktree manager since we are
        # wrapping it in GitWorktreeMultiManager.execute_concurrently
        graph_evaluator = SubAgent(
            llm=self.llm,
            system_prompt="You are an isolated SubAgent responsible for strictly evaluating responses and plans.",
            use_isolation=False
        )

        async def evaluation_task(worktree_path: str) -> str:
            # Inject worktree path context
            full_context = context + f"\nIsolated Worktree Context: {worktree_path}"
            return await graph_evaluator.execute(task=task, context=full_context, temperature=0.1)

        results = await self.worktree_manager.execute_concurrently([evaluation_task])
        return results
