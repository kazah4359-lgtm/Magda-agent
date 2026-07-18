import logging
import re
from typing import Optional, List, Dict, Any
from magda_agent.llm_client import LLMClient

class RPCPayloadCompressor:
    """
    Compresses large contexts dynamically for spawned subagents.
    Optimizes token usage based on task relevance while strictly preserving
    critical constraints and instructions.
    """

    def __init__(self, llm: LLMClient) -> None:
        """
        Initializes the RPCPayloadCompressor.

        Args:
            llm: Language Model client to be used for summarization.
        """
        self.llm = llm

    def _extract_critical_constraints(self, text: str) -> List[str]:
        """
        Extracts sentences or lines that contain high-priority constraint keywords
        to guarantee they are not lost during compression.

        Args:
            text: Raw input text.

        Returns:
            A list of extracted constraint strings.
        """
        keywords = ["must", "never", "strict", "required", "limit", "critical", "mandatory", "rule", "goal", "constraint"]
        pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)

        constraints: List[str] = []
        # Split by newline first, then by sentence markers if needed
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            if pattern.search(line):
                constraints.append(line)
        return list(dict.fromkeys(constraints))  # Deduplicate while preserving order

    async def compress_payload(self, payload: Dict[str, Any], max_length: int = 2000) -> Dict[str, Any]:
        """
        Selectively compresses context inside payload, retaining critical constraints.

        Args:
            payload: The dictionary representing the RPC dispatch payload.
            max_length: The maximum allowed length of the context. Triggers compression if exceeded.

        Returns:
            The optimized payload dictionary.
        """
        context = payload.get("context", "")
        task = payload.get("task", "")

        if len(context) <= max_length:
            logging.info("Context is within limits. Skipping compression.")
            return payload

        logging.info(f"Context length ({len(context)}) exceeds max_length ({max_length}). Compressing based on task relevance.")

        # Identify critical constraints as a backup/validation check
        constraints = self._extract_critical_constraints(context)

        prompt = (
            f"You are a context compression engine optimizing tokens for a spawned subagent.\n"
            f"Your job is to compress the parent context below so that it fits within budget, "
            f"keeping ONLY the information relevant to the subagent's assigned task, while "
            f"STRICTLY preserving all critical constraints and rules (e.g., MUST, REQUIRED, NEVER, LIMIT, etc.).\n\n"
            f"--- ASSIGNED SUBAGENT TASK ---\n"
            f"{task}\n\n"
            f"--- PARENT CONTEXT TO COMPRESS ---\n"
            f"{context}\n\n"
            f"Instructions:\n"
            f"1. Extract and retain all key constraints, rules, and mandatory bounds from the Parent Context.\n"
            f"2. Summarize or remove parts of the Parent Context that are irrelevant or secondary to the Assigned Subagent Task.\n"
            f"3. Do not lose critical formatting, identifiers, or keys needed to execute the task.\n"
            f"4. The output must be concise and optimized for token savings."
        )

        messages = [
            {"role": "system", "content": "You are an advanced token compression system specializing in task-relevant summarization and constraint preservation."},
            {"role": "user", "content": prompt}
        ]

        try:
            compressed = await self.llm.chat_completion(messages, temperature=0.2)
            compressed_str = compressed.strip()

            if not compressed_str:
                return payload

            # Double-check if constraints are preserved. If any constraint from original is missing, append it as metadata/safety.
            missing_constraints = []
            for const in constraints:
                # Basic check: is a substantial portion of the constraint present?
                # Lowercase clean check
                const_clean = re.sub(r"\s+", " ", const.lower()).strip()
                compressed_clean = re.sub(r"\s+", " ", compressed_str.lower()).strip()
                if const_clean not in compressed_clean:
                    missing_constraints.append(const)

            if missing_constraints:
                logging.info(f"Re-injecting {len(missing_constraints)} missing critical constraints.")
                constraint_header = "\n\n--- PRESERVED CRITICAL CONSTRAINTS ---\n" + "\n".join(f"- {c}" for c in missing_constraints)
                compressed_str += constraint_header

            payload["context"] = compressed_str
            return payload
        except Exception as e:
            logging.error(f"Failed to compress payload dynamically: {e}")
            return payload
