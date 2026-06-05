import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from magda_agent.subconsciousness.reflection import Subconsciousness
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.memory.semantic import SemanticMemory
from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry

@pytest.fixture
def mock_semantic_memory():
    sm = MagicMock(spec=SemanticMemory)
    # Simulate an existing fact
    sm.search_facts.return_value = [{"id": "123", "text": "The sky is blue", "metadata": {}}]
    return sm

@pytest.fixture
def mock_memory_system():
    ms = MagicMock(spec=MemorySystem)
    ms.short_term = [MemoryEntry(content="Someone told me the sky is red.", importance=0.8, emotional_state=None)]
    return ms

@pytest.fixture
def mock_llm_conflict():
    llm = MagicMock(spec=LLMClient)

    # Needs to return 2 things: the reflection response, then the conflict response.
    # We will use side_effect
    async def mock_chat_completion(messages, **kwargs):
        if "detect semantic memory contradictions" in messages[0]["content"]:
            return json.dumps({
                "conflict": True,
                "conflicting_id": "123",
                "strategy": "newer_wins",
                "resolved_fact": "The sky is red"
            })
        else:
            return json.dumps({
                "summary": "Reflection summary",
                "lessons": [],
                "anti_patterns": [],
                "proposed_tasks": [],
                "new_facts": ["The sky is red"],
                "pad_adjustment": {"p": 0.0, "a": 0.0, "d": 0.0}
            })

    llm.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    return llm

@pytest.mark.asyncio
async def test_subconsciousness_conflict_detection(mock_llm_conflict, mock_memory_system, mock_semantic_memory):
    emotions = EmotionalEngine()

    subconsciousness = Subconsciousness(
        llm=mock_llm_conflict,
        emotions=emotions,
        memory=mock_memory_system,
        semantic_memory=mock_semantic_memory,
        interval=1
    )

    await subconsciousness.reflect()

    # Check that search_facts was called with the new fact
    mock_semantic_memory.search_facts.assert_called_with("The sky is red", top_k=3)

    # Check that the conflict resolution deleted the old fact and stored the new one
    mock_semantic_memory.delete_fact.assert_called_with("123")
    mock_semantic_memory.store_fact.assert_called_with("The sky is red")
