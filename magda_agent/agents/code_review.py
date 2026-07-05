import json
import logging
from typing import List, Dict, Optional

from magda_agent.agents.sub_agent import SubAgent
from magda_agent.llm_client import LLMClient

class CodeReviewWorker(SubAgent):
    """
    SubAgent specialized for reviewing code pull requests.
    Inspired by Claude Agent SDK code review patterns.
    """
    def __init__(self, llm: LLMClient, system_prompt: Optional[str] = None, use_isolation: bool = False):
        """
        Initializes the CodeReviewWorker.

        Args:
            llm (LLMClient): The LLM client to use for code review.
            system_prompt (Optional[str]): Custom system prompt for the worker.
            use_isolation (bool): Whether to use isolated git worktrees.
        """
        default_prompt = (
            "You are an expert software engineer reviewing a pull request. "
            "Analyze the provided git diff carefully. Identify any bugs, security vulnerabilities, "
            "code style issues, or missing tests. "
            "You must output your review exactly as a JSON array of objects, where each object "
            'has a "comment_id" and a "reply" string field. '
            'Example: [{"comment_id": "file.py:10", "reply": "This variable is unused."}]'
        )
        prompt = system_prompt or default_prompt
        super().__init__(llm=llm, system_prompt=prompt, use_isolation=use_isolation)

    async def review_pr(self, pr_diff: str) -> List[Dict[str, str]]:
        """
        Reviews a PR diff and returns structured comments.

        Args:
            pr_diff (str): The git diff string of the pull request.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing 'comment_id' and 'reply'.
        """
        logging.info(f"CodeReviewWorker starting review for diff of length {len(pr_diff)}")

        context = "Please review the following PR diff:\n\n"
        task = pr_diff

        # We use the parent's execute method to handle the LLM interaction
        result_text = await self.execute(task=task, context=context)

        # Attempt to parse the JSON output from the LLM
        try:
            # We will clean up the result in case there are markdown code block ticks
            clean_text = result_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            comments = json.loads(clean_text)

            if not isinstance(comments, list):
                logging.error("CodeReviewWorker LLM output was not a JSON list.")
                return []

            valid_comments = []
            for comment in comments:
                if isinstance(comment, dict) and "comment_id" in comment and "reply" in comment:
                    valid_comments.append({
                        "comment_id": str(comment["comment_id"]),
                        "reply": str(comment["reply"])
                    })
                else:
                    logging.warning(f"CodeReviewWorker skipped invalid comment format: {comment}")

            logging.info(f"CodeReviewWorker generated {len(valid_comments)} comments.")
            return valid_comments

        except json.JSONDecodeError as e:
            logging.error(f"CodeReviewWorker failed to parse JSON: {e}\nRaw output: {result_text}")
            return []
