import pytest
from unittest.mock import AsyncMock

from magda_agent.emotions.engine import PADState
from magda_agent.llm_client import LLMClient
from magda_agent.memory.working import MemoryEntry, WorkingMemory
from magda_agent.memory.virtual_compression_trigger_v3 import MemGPTVirtualCompressionTriggerV3


@pytest.mark.asyncio
async def test_trigger_under_threshold() -> None:
    llm_mock = AsyncMock(spec=LLMClient)
    trigger = MemGPTVirtualCompressionTriggerV3(llm=llm_mock, token_threshold=2000)

    state = PADState(0.1, 0.2, 0.3)
    e1 = MemoryEntry("short context 1", importance=0.5, emotional_state=state, user_id=1)
    e2 = MemoryEntry("short context 2", importance=0.5, emotional_state=state, user_id=1)
    entries = [e1, e2]

    assert trigger.should_trigger(entries) is False

    was_compressed, result = await trigger.check_and_compress(entries)

    assert was_compressed is False
    assert result == entries
    assert trigger.trigger_count == 1
    assert trigger.total_compressions == 0
    llm_mock.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_trigger_over_threshold_with_llm() -> None:
    llm_mock = AsyncMock(spec=LLMClient)
    llm_mock.chat_completion.return_value = "Compressed dialogue summary."

    trigger = MemGPTVirtualCompressionTriggerV3(llm=llm_mock, token_threshold=10)

    state = PADState(0.1, 0.2, 0.3)
    e1 = MemoryEntry("word1 word2 word3 word4 word5", importance=0.5, emotional_state=state, user_id=1)
    e2 = MemoryEntry("word6 word7 word8 word9 word10", importance=0.6, emotional_state=state, user_id=1)
    e3 = MemoryEntry("word11 word12 word13 word14 word15", importance=0.7, emotional_state=state, user_id=1)
    e4 = MemoryEntry("word16 word17 word18 word19 word20", importance=0.8, emotional_state=state, user_id=1)
    entries = [e1, e2, e3, e4]

    assert trigger.should_trigger(entries) is True

    was_compressed, result = await trigger.check_and_compress(entries)

    assert was_compressed is True
    assert len(result) == 3
    assert result[0].content == "Compressed dialogue summary."
    assert result[1] == e3
    assert result[2] == e4
    assert trigger.trigger_count == 1
    assert trigger.total_compressions == 1
    assert trigger.last_compressed_tokens > 10
    llm_mock.chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_trigger_with_working_memory() -> None:
    llm_mock = AsyncMock(spec=LLMClient)
    llm_mock.chat_completion.return_value = "Summary of early entries."

    trigger = MemGPTVirtualCompressionTriggerV3(llm=llm_mock, token_threshold=10)
    wm = WorkingMemory()

    state = PADState(0.1, 0.2, 0.3)
    await wm.add(MemoryEntry("entry one has five words here", importance=0.5, emotional_state=state, user_id=1))
    await wm.add(MemoryEntry("entry two has five words here", importance=0.5, emotional_state=state, user_id=1))
    await wm.add(MemoryEntry("entry three has five words here", importance=0.5, emotional_state=state, user_id=1))

    assert trigger.should_trigger(wm, user_id=1) is True

    was_compressed, result = await trigger.check_and_compress(wm, user_id=1)

    assert was_compressed is True
    assert len(wm.get_entries(user_id=1)) == len(result)
    assert result[0].content == "Summary of early entries."


@pytest.mark.asyncio
async def test_trigger_empty_entries() -> None:
    trigger = MemGPTVirtualCompressionTriggerV3(token_threshold=10)
    assert trigger.should_trigger([]) is False

    was_compressed, result = await trigger.check_and_compress([])
    assert was_compressed is False
    assert result == []
