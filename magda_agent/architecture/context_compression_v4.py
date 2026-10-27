"""
Claude Agent Teams Context Compression V4.

This module provides the ClaudeSubagentContextCompressionV4 class (and ContextCompressionV4 alias)
which adds context compression hooks for subagent spawners to format, condense, and pass
hierarchical context down the subagent hierarchy while strictly retaining critical instructions,
rules, and safety constraints.
"""

import logging
import re
from typing import Optional, List, Dict, Any, Union
from magda_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ClaudeSubagentContextCompressionV4:
    """
    ClaudeSubagentContextCompressionV4 manages context compression and hierarchical formatting
    for spawned subagents in multi-agent agent teams.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        max_context_length: int = 2000,
        max_messages: int = 5
    ) -> None:
        """
        Initialize the ClaudeSubagentContextCompressionV4.

        Args:
            llm: Optional LLMClient instance for dynamic summarization.
            max_context_length: Maximum allowed character length for context strings before triggering compression.
            max_messages: Maximum message turns to keep when compressing message history list.
        """
        self.llm = llm
        self.max_context_length = max_context_length
        self.max_messages = max_messages

    def extract_critical_constraints(self, text: str) -> List[str]:
        """
        Extract sentences or lines containing high-priority constraint keywords
        so they are preserved during compression.

        Args:
            text: Input context string.

        Returns:
            List of unique constraint lines.
        """
        keywords = [
            "must", "never", "strict", "required", "limit", "critical",
            "mandatory", "rule", "goal", "constraint", "shall", "always",
            "do not", "safety", "forbidden"
        ]
        pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)

        constraints: List[str] = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            if pattern.search(line):
                constraints.append(line)

        # Deduplicate while preserving order
        return list(dict.fromkeys(constraints))

    def format_hierarchical_context(
        self,
        parent_goal: str,
        subtask: str,
        history: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Format hierarchical context for a subagent incorporating parent goal, subtask,
        and conversation history.

        Args:
            parent_goal: Overall parent goal or mission objective.
            subtask: Specific subtask assigned to the subagent.
            history: List of message dictionaries.
            metadata: Optional additional metadata or environmental factors.

        Returns:
            List of formatted message dictionaries.
        """
        system_content = f"Parent Goal: {parent_goal}\nAssigned Subtask: {subtask}"
        if metadata:
            meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            system_content += f"\nContext Metadata:\n{meta_str}"

        formatted_messages = [
            {"role": "system", "content": system_content}
        ]

        for msg in history:
            if msg.get("role") != "system":
                formatted_messages.append(msg)

        return formatted_messages

    async def compress_context(
        self,
        context: str,
        task: str = "",
        max_length: Optional[int] = None
    ) -> str:
        """
        Compress context string based on task relevance and token limits.

        Args:
            context: Context string to compress.
            task: Subagent task description.
            max_length: Optional max length threshold override.

        Returns:
            Condensed context string.
        """
        limit = max_length if max_length is not None else self.max_context_length

        if len(context) <= limit:
            return context

        constraints = self.extract_critical_constraints(context)

        if self.llm is not None:
            prompt = (
                f"You are an AGI context compression module for subagents.\n"
                f"Compress the context below so it fits under target size limits.\n"
                f"Keep ONLY details relevant to the subagent task, and STRICTLY preserve\n"
                f"all critical safety constraints, rules, and mandatory requirements.\n\n"
                f"--- SUBAGENT TASK ---\n{task}\n\n"
                f"--- CONTEXT TO COMPRESS ---\n{context}\n\n"
                f"Instructions:\n"
                f"1. Retain all rules, constraints, identifiers, and bounds.\n"
                f"2. Omit irrelevant details.\n"
                f"3. Return only the condensed text."
            )
            messages = [
                {"role": "system", "content": "You are a specialized subagent context compressor."},
                {"role": "user", "content": prompt}
            ]
            try:
                summary = await self.llm.chat_completion(messages, temperature=0.2)
                compressed_str = summary.strip()

                if compressed_str:
                    # Check for missing constraints and re-inject if missing
                    missing = []
                    for const in constraints:
                        c_clean = re.sub(r"\s+", " ", const.lower()).strip()
                        comp_clean = re.sub(r"\s+", " ", compressed_str.lower()).strip()
                        if c_clean not in comp_clean:
                            missing.append(const)

                    if missing:
                        reinject_str = "\n\n--- PRESERVED CRITICAL CONSTRAINTS ---\n" + "\n".join(f"- {c}" for c in missing)
                        compressed_str += reinject_str

                    return compressed_str
            except Exception as e:
                logger.error(f"Error during LLM context compression: {e}")

        # Fallback compression logic when LLM is unavailable or fails
        truncated = context[:limit]
        if constraints:
            reinject_str = "\n\n--- PRESERVED CRITICAL CONSTRAINTS ---\n" + "\n".join(f"- {c}" for c in constraints)
            # Ensure total length respects budget if possible or append constraints
            if len(truncated) + len(reinject_str) > limit and limit > len(reinject_str):
                truncated = truncated[: limit - len(reinject_str)]
            truncated += reinject_str

        return truncated

    async def compress_messages(
        self,
        messages: List[Dict[str, Any]],
        task: str = "",
        max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Compress a message history list by keeping system messages and recent turns
        while summarizing older turns.

        Args:
            messages: List of message dicts.
            task: Subagent task description.
            max_messages: Maximum allowed message count override.

        Returns:
            Compressed list of message dicts.
        """
        limit = max_messages if max_messages is not None else self.max_messages

        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        if len(non_system_messages) <= limit:
            return messages

        older_messages = non_system_messages[:-limit]
        recent_messages = non_system_messages[-limit:]

        older_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in older_messages)

        if self.llm is not None:
            prompt = (
                f"Summarize the following conversation history for a subagent executing task: '{task}'.\n"
                f"Focus on key technical guidelines, decisions, and constraints.\n\n"
                f"--- CONVERSATION HISTORY ---\n{older_text}"
            )
            compress_prompt = [
                {"role": "system", "content": "You are a context compression assistant for subagents."},
                {"role": "user", "content": prompt}
            ]
            try:
                summary = await self.llm.chat_completion(compress_prompt, temperature=0.2)
                summary_msg = {
                    "role": "system",
                    "content": f"[SYSTEM: Compressed Context Summary: {summary.strip()}]"
                }
                return system_messages + [summary_msg] + recent_messages
            except Exception as e:
                logger.error(f"Error during message compression: {e}")

        # Fallback message summary
        constraints = self.extract_critical_constraints(older_text)
        summary_text = f"[SYSTEM: Compressed {len(older_messages)} historical turns]"
        if constraints:
            summary_text += " Key constraints: " + "; ".join(constraints)

        summary_msg = {
            "role": "system",
            "content": summary_text
        }

        return system_messages + [summary_msg] + recent_messages

    async def compress_payload(
        self,
        payload: Dict[str, Any],
        task: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compress subagent dispatch payload dictionary containing 'context', 'messages', or 'task'.

        Args:
            payload: Payload dictionary.
            task: Task description override.
            max_length: Max context length override.

        Returns:
            Compressed payload dictionary.
        """
        result = dict(payload)
        subtask = task or result.get("task", "")

        if "context" in result and isinstance(result["context"], str):
            result["context"] = await self.compress_context(
                result["context"],
                task=subtask,
                max_length=max_length
            )

        if "messages" in result and isinstance(result["messages"], list):
            result["messages"] = await self.compress_messages(
                result["messages"],
                task=subtask
            )

        return result

    async def compress_hierarchical_context(
        self,
        parent_goal: str,
        subtask: str,
        history: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Format and compress hierarchical subagent context in a single convenience call.

        Args:
            parent_goal: Parent objective.
            subtask: Subtask description.
            history: Conversation history.
            metadata: Context metadata.
            max_messages: Max messages to retain.

        Returns:
            Formatted and compressed message list.
        """
        formatted = self.format_hierarchical_context(
            parent_goal=parent_goal,
            subtask=subtask,
            history=history,
            metadata=metadata
        )
        return await self.compress_messages(
            messages=formatted,
            task=subtask,
            max_messages=max_messages
        )


# Alias for backward compatibility and clean naming
ContextCompressionV4 = ClaudeSubagentContextCompressionV4
