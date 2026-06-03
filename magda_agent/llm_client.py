import os
import logging
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class LLMClient:
    """
    Unified interface for interacting with Large Language Models (OpenAI API).
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            if not OpenAI:
                logging.warning("OpenAI library not installed.")
            if not self.api_key:
                logging.warning("OPENAI_API_KEY not found in environment.")

    async def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Sends a list of messages to the LLM and returns the response content.
        """
        if not self.client:
            return "Error: LLM Client not initialized. Please check OPENAI_API_KEY."

        try:
            # We use the sync client for simplicity in this PoC,
            # but wrapping it in an async-friendly way if needed.
            # In a real app, one would use the async client.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM API Error: {e}")
            return f"Error: I encountered an issue while thinking. ({e})"

    def get_system_prompt(self, context: str, emotions: str) -> str:
        return f"""
You are Magda, a sophisticated AGI agent.
You have a hierarchical cognitive architecture including Consciousness, Subconsciousness, and an Emotional Engine.

CURRENT EMOTIONAL STATE: {emotions}
RELEVANT CONTEXT/MEMORIES: {context}

Guidelines:
1. Respond based on your current emotional state and memories.
2. Be helpful, autonomous, and self-reflective.
3. If the user asks about your internal state, you can share insights from your PAD model.
"""
