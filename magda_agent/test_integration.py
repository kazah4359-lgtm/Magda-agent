import asyncio
import os
from magda_agent.memory.long_term_memory import LongTermMemory
from magda_agent.emotions.emotional_engine import EmotionalEngine
from magda_agent.consciousness.engine import Consciousness
from magda_agent.subconsciousness.reflexion import Subconsciousness

async def test_integration():
    print("--- Starting Integration Test ---")

    # 1. Initialize modules
    memory = LongTermMemory("test_memory.json")
    emotions = EmotionalEngine()
    consciousness = Consciousness(memory, emotions)
    subconsciousness = Subconsciousness(memory, emotions)

    # 2. Test initial state
    status = consciousness.get_status()
    print(f"Initial Mood: {status['mood']}")
    assert status['mood'] == "Neutral"

    # 3. Process some input
    print("User: Hello Magdalina!")
    response = await consciousness.process_input("Hello Magdalina!")
    print(f"Magdalina: {response}")

    # 4. Check emotional change
    status = consciousness.get_status()
    print(f"Mood after greeting: {status['mood']}")
    print(f"Emotional State: {status['emotional_state']}")
    assert status['emotional_state']['arousal'] > 0

    # 5. Test memory retrieval
    print("User: Tell me about Magdalina.")
    response = await consciousness.process_input("Tell me about Magdalina.")
    print(f"Magdalina: {response}")
    assert "magdalina" in response.lower() or "remember" in response.lower()

    # 6. Test subconscious reflection
    print("Running subconscious reflection...")
    subconsciousness.memory.consolidate()
    subconsciousness.emotions.decay()

    new_status = consciousness.get_status()
    print(f"Mood after decay: {new_status['mood']}")

    # Cleanup
    if os.path.exists("test_memory.json"):
        os.remove("test_memory.json")

    print("--- Integration Test Passed! ---")

if __name__ == "__main__":
    asyncio.run(test_integration())
