import pytest
import asyncio
from magda_agent.memory.compression import ContextCompressorSelective
from magda_agent.memory.working import MemoryEntry
from magda_agent.emotions.engine import PADState
from typing import List, Dict, Any

class MockLLMClient:
    """Mock LLM client for testing without making real API calls."""
    async def chat_completion(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Mocks the chat completion endpoint."""
        return "Compressed summary."

@pytest.mark.asyncio
async def test_compress_entries() -> None:
    """Tests that entries are correctly compressed using the mock LLM."""
    llm = MockLLMClient()
    compressor = ContextCompressorSelective(llm_client=llm)
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("User says hello.", 0.5, state, tags=["greeting"], user_id=1)
    e2 = MemoryEntry("User says goodbye.", 0.6, state, tags=["farewell"], user_id=1)

    result = await compressor.compress_entries([e1, e2])
    assert result.content == "Compressed summary."
    assert result.importance == 0.55
    assert result.user_id == 1
    assert "greeting" in result.tags
    assert "farewell" in result.tags

def test_selective_retrieval() -> None:
    """Tests the selective retrieval of memory entries based on keyword queries."""
    compressor = ContextCompressorSelective()
    state = PADState(0, 0, 0)

    e1 = MemoryEntry("This is about apples and oranges.", 0.5, state)
    e2 = MemoryEntry("This is about cars and bikes.", 0.5, state)

    results = compressor.selective_retrieval([e1, e2], "apples")
    assert len(results) == 1
    assert results[0].content == "This is about apples and oranges."

    results_empty = compressor.selective_retrieval([e1, e2], "bananas")
    assert len(results_empty) == 0


@pytest.mark.asyncio
async def test_compress_workflow() -> None:
    """Tests that compress_workflow respects token limits and falls back correctly if needed."""
    llm = MockLLMClient()
    compressor = ContextCompressorSelective(llm_client=llm)

    # Test within limit
    short_context = "This is a short context."
    result = await compressor.compress_workflow(short_context, 100)
    assert result == short_context

    # Test exceeding limit with LLM
    long_context = "A" * 500  # 500 characters
    result_llm = await compressor.compress_workflow(long_context, 10) # 10 tokens * 4 = 40 chars limit
    assert result_llm == "Compressed summary."

    # Test exceeding limit without LLM (fallback to truncation)
    compressor_no_llm = ContextCompressorSelective(llm_client=None)
    result_no_llm = await compressor_no_llm.compress_workflow(long_context, 10)
    assert result_no_llm.endswith("... [TRUNCATED]")
    assert len(result_no_llm) == 40 + len("... [TRUNCATED]")
