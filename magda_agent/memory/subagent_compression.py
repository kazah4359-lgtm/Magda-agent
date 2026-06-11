import logging
from typing import Optional
from magda_agent.llm_client import LLMClient

class SubagentContextCompressor:
    """
    Compresses large contexts for subagents to optimize token usage.
    Inspired by Claude Agent SDK context compression.
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def compress_context(self, context: str, max_length: int = 2000) -> str:
        """
        Compresses the context if it exceeds the max_length.
        """
        if len(context) <= max_length:
            return context

        logging.info(f"Compressing subagent context of length {len(context)}")
        prompt = "Please summarize the following context concisely, preserving all essential details and constraints for a sub-agent task:\n\n" + context

        messages = [
            {"role": "system", "content": "You are a context compression engine. Return only the compressed context."},
            {"role": "user", "content": prompt}
        ]

        try:
            compressed = await self.llm.chat_completion(messages, temperature=0.3)
            return compressed.strip()
        except Exception as e:
            logging.error(f"Failed to compress context: {e}")
            return context  # Fallback to original context
