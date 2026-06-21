import os
import logging
import asyncio
from typing import List, Dict, Any, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

class LLMClient:
    """
    Unified interface for interacting with Large Language Models (OpenAI API).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = None

        if AsyncOpenAI and self.api_key:
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = AsyncOpenAI(**client_kwargs)
        else:
            if not AsyncOpenAI:
                logging.warning("OpenAI library (Async) not installed.")
            if not self.api_key:
                logging.warning("OPENAI_API_KEY not found in environment.")

    async def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Sends a list of messages to the LLM and returns the response content asynchronously.
        """
        if not self.client:
            return "Error: LLM Client not initialized. Please check OPENAI_API_KEY."

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM API Error: {e}")
            return f"Error: I encountered an issue while thinking. ({e})"

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generates a response from a single prompt.
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, temperature=temperature)

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
