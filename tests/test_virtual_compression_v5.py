import asyncio
import pytest
from unittest.mock import AsyncMock

from magda_agent.emotions.engine import PADState
from magda_agent.llm_client import LLMClient
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory
from magda_agent.memory.virtual_compression_v5 import (
    MemGPTVirtualContextSemanticCompressorV5,
    MemGPTVirtualCompressionV5,
)
from magda_agent.memory.working import MemoryEntry, WorkingMemory


@pytest.fixture
def episodic_mem():
    mem = EpisodicMemory(persist_directory=":memory:")
    try:
        mem.client.delete_collection("episodic_memory")
    except Exception:
        pass
    mem.collection = mem.client.get_or_create_collection("episodic_memory")
    return mem


@pytest.fixture
def semantic_mem():
    mem = SemanticMemory(persist_directory=":memory:")
    try:
        mem.client.delete_collection("semantic_memory")
    except Exception:
        pass
    mem.collection = mem.client.get_or_create_collection("semantic_memory")
    return mem


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMClient)
    llm.generate = AsyncMock(
        return_value="- User prefers Python over TypeScript for backend\n- User is building Magda AGI framework"
    )
    return llm


@pytest.mark.asyncio
async def test_compress_episodic_to_semantic_success(episodic_mem, semantic_mem, mock_llm):
    compressor = MemGPTVirtualContextSemanticCompressorV5(
        episodic_memory=episodic_mem,
        semantic_memory=semantic_mem,
        llm_client=mock_llm,
        batch_size=5,
        min_events_threshold=2,
    )

    uid = 998877
    episodic_mem.store_event("User said they prefer Python for backend services", user_id=uid)
    episodic_mem.store_event("User discussed architecture of Magda AGI framework", user_id=uid)
    episodic_mem.store_event("User discussed deployment to Linux sandbox", user_id=uid)

    facts = await compressor.compress_episodic_to_semantic(user_id=uid)

    assert len(facts) == 2
    assert "User prefers Python over TypeScript for backend" in facts
    assert "User is building Magda AGI framework" in facts

    recalled = semantic_mem.recall_facts("Python", user_id=uid)
    assert len(recalled) > 0

    remaining_events = episodic_mem.get_all_events(user_id=uid, include_decayed=False)
    assert len(remaining_events) == 0

    assert compressor.total_compressions == 1
    assert compressor.total_facts_extracted == 2
    assert compressor.total_events_compacted == 3


@pytest.mark.asyncio
async def test_compress_below_threshold(episodic_mem, semantic_mem, mock_llm):
    compressor = MemGPTVirtualContextSemanticCompressorV5(
        episodic_memory=episodic_mem,
        semantic_memory=semantic_mem,
        llm_client=mock_llm,
        batch_size=5,
        min_events_threshold=3,
    )

    uid = 998878
    episodic_mem.store_event("Single event in episodic memory", user_id=uid)

    facts = await compressor.compress_episodic_to_semantic(user_id=uid)

    assert facts == []
    mock_llm.generate.assert_not_called()
    assert compressor.total_compressions == 0


@pytest.mark.asyncio
async def test_compress_with_working_memory_clearing(episodic_mem, semantic_mem, mock_llm):
    working_mem = WorkingMemory()
    state = PADState(0.1, 0.2, 0.3)
    uid = 998879
    await working_mem.add(MemoryEntry("entry 1", importance=0.5, emotional_state=state, user_id=uid))
    await working_mem.add(MemoryEntry("entry 2", importance=0.5, emotional_state=state, user_id=uid))
    await working_mem.add(MemoryEntry("entry 3", importance=0.5, emotional_state=state, user_id=uid))
    await working_mem.add(MemoryEntry("entry 4", importance=0.5, emotional_state=state, user_id=uid))

    compressor = MemGPTVirtualCompressionV5(
        episodic_memory=episodic_mem,
        semantic_memory=semantic_mem,
        working_memory=working_mem,
        llm=mock_llm,
        min_events_threshold=2,
    )

    episodic_mem.store_event("Event A", user_id=uid)
    episodic_mem.store_event("Event B", user_id=uid)

    facts = await compressor.compress_episodic_to_semantic(user_id=uid)

    assert len(facts) == 2
    remaining_wm = working_mem.get_entries(user_id=uid)
    assert len(remaining_wm) == 2


@pytest.mark.asyncio
async def test_llm_error_handling(episodic_mem, semantic_mem):
    mock_error_llm = AsyncMock(spec=LLMClient)
    mock_error_llm.generate = AsyncMock(return_value="Error: Connection failed")

    compressor = MemGPTVirtualContextSemanticCompressorV5(
        episodic_memory=episodic_mem,
        semantic_memory=semantic_mem,
        llm_client=mock_error_llm,
        min_events_threshold=2,
    )

    uid = 998880
    episodic_mem.store_event("Event A", user_id=uid)
    episodic_mem.store_event("Event B", user_id=uid)

    facts = await compressor.compress_episodic_to_semantic(user_id=uid)

    assert facts == []
    assert compressor.total_compressions == 0


@pytest.mark.asyncio
async def test_background_loop_lifecycle(episodic_mem, semantic_mem, mock_llm):
    compressor = MemGPTVirtualContextSemanticCompressorV5(
        episodic_memory=episodic_mem,
        semantic_memory=semantic_mem,
        llm_client=mock_llm,
        min_events_threshold=2,
    )

    uid = 998881
    episodic_mem.store_event("Event 1", user_id=uid)
    episodic_mem.store_event("Event 2", user_id=uid)

    task = compressor.start_background_loop(interval_seconds=0.05, user_id=uid)
    await asyncio.sleep(0.12)
    await compressor.stop_background_loop()

    assert task.done()
    assert compressor.total_compressions >= 1
