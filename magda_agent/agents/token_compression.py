import logging
import re
from typing import Optional, List, Dict, Any
from magda_agent.llm_client import LLMClient

class SubagentTokenCompressor:
    """
    Compresses large contexts dynamically for spawned subagents.
    Optimizes token usage based on task relevance while strictly preserving
    critical constraints and instructions.
    """

    def __init__(self, llm: LLMClient) -> None:
        """
        Initializes the SubagentTokenCompressor.

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
        keywords = ["must", "never", "strict", "required", "limit", "critical", "mandatory", "rule"]
        pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)

        constraints: List[str] = []
        # Split by newline first, then by sentence markers if needed
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            if pattern.search(line):
                constraints.append(line)
        return list(dict.fromkeys(constraints))  # Deduplicate while preserving order

    async def compress_context(self, context: str, task: str, max_length: int = 2000) -> str:
        """
        Selectively compresses context based on task relevance, retaining critical constraints.

        Args:
            context: The raw context string to compress.
            task: The subagent task description.
            max_length: The maximum allowed length of the context. Triggers compression if exceeded.

        Returns:
            The compressed context string.
        """
        if len(context) <= max_length:
            logging.info("Context is within limits. Skipping compression.")
            return f"Parent Context:\n{context}\n\nAssigned Task:\n{task}"

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
                return context[:max_length]

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

            return compressed_str
        except Exception as e:
            logging.error(f"Failed to compress context dynamically: {e}")
            return context[:max_length]

    async def compress_messages(self, messages: List[Dict[str, Any]], task: str, max_messages: int = 5) -> List[Dict[str, Any]]:
        """
        Compresses a structured list of conversation messages by summarizing older messages
        and preserving only the most recent/relevant ones.

        Args:
            messages: List of message dictionaries, each having 'role' and 'content'.
            task: The assigned task of the subagent.
            max_messages: Target number of messages to keep in the history.

        Returns:
            An optimized list of messages.
        """
        if len(messages) <= max_messages:
            return messages

        logging.info(f"Compressing message history of {len(messages)} items down to {max_messages}.")

        # Keep system prompt and the most recent messages untouched
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_assistant_messages = [msg for msg in messages if msg.get("role") != "system"]

        if len(user_assistant_messages) <= max_messages:
            return messages

        # Messages to compress/summarize
        to_compress = user_assistant_messages[:-max_messages]
        retained = user_assistant_messages[-max_messages:]

        # Serialize history for compression
        history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in to_compress])

        prompt = (
            f"The following is a list of older conversational history entries.\n"
            f"Please compress and summarize this history into a single concise paragraph. "
            f"Focus especially on any constraints, preferences, or technical details relevant to the subagent task.\n\n"
            f"--- SUBAGENT TASK ---\n"
            f"{task}\n\n"
            f"--- OLD CONVERSATION HISTORY ---\n"
            f"{history_text}"
        )

        compress_msgs = [
            {"role": "system", "content": "You are a context compression assistant. Concisely summarize old message history relevant to the current task."},
            {"role": "user", "content": prompt}
        ]

        try:
            summary = await self.llm.chat_completion(compress_msgs, temperature=0.2)
            summary_str = summary.strip()

            summary_message = {
                "role": "system",
                "content": f"[SYSTEM: Compressed Summary of Old History: {summary_str}]"
            }

            return system_messages + [summary_message] + retained
        except Exception as e:
            logging.error(f"Failed to compress message history: {e}")
            # Fallback: simple truncation
            return system_messages + retained
