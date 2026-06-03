import asyncio
import logging
import sys
import pytest
from magda_agent.llm_client import LLMClient
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills import initialize_skills
from magda_agent.consciousness.core import Consciousness
from magda_agent.subconsciousness.reflection import Subconsciousness

@pytest.mark.asyncio
async def test_integration():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Magda Agent Integration Test...")

    # Mocking LLM client to avoid API calls and costs during tests
    class MockLLMClient(LLMClient):
        async def chat_completion(self, messages, temperature=0.7):
            return "This is a mock response from Magda."

    llm = MockLLMClient(api_key="test_key")
    emotions = EmotionalEngine()
    memory = MemorySystem()
    skills = initialize_skills()

    consciousness = Consciousness(llm, emotions, memory, skills)
    subconsciousness = Subconsciousness(llm, emotions, memory, interval=1)

    # 1. Test Consciousness processing
    logging.info("Testing Consciousness...")
    response = await consciousness.process_input("Hello, how are you?")
    assert "mock response" in response.lower()
    assert len(memory.short_term) > 0
    logging.info("Consciousness test passed.")

    # 2. Test Emotional State
    logging.info(f"Current Emotion: {emotions.get_emotion_label()}")
    assert emotions.state.arousal != 0.0

    # 3. Test Subconsciousness reflection
    logging.info("Testing Subconsciousness...")
    await subconsciousness.reflect()
    assert any("Subconscious reflection" in m.content for m in memory.short_term)
    logging.info("Subconsciousness test passed.")

    # 4. Test Skills Registry
    logging.info("Testing Skills...")
    result = skills.execute_skill("programmer", code="print('Hello World')")
    assert "Hello World" in result
    logging.info("Skills test passed.")

    logging.info("Integration Test Successful!")

if __name__ == "__main__":
    asyncio.run(test_integration())
