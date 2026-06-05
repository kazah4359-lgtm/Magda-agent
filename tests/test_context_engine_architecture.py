import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.context_engine import ContextEngine, ContextPlugin
from magda_agent.memory.working import WorkingMemory, MemoryEntry
from magda_agent.thalamus.router import Thalamus
from magda_agent.emotions.engine import PADState

@pytest.fixture
def mock_plugin():
    plugin = ContextPlugin()
    plugin.pre_process = AsyncMock(side_effect=lambda x: f"pre_{x}")
    plugin.compress = AsyncMock(side_effect=lambda x: [x[0]]) # Mock compressing list of 2 down to 1
    plugin.post_process = AsyncMock(side_effect=lambda x: [f"post_{item}" for item in x])
    return plugin

@pytest.fixture
def context_engine(mock_plugin):
    engine = ContextEngine()
    engine.register_plugin(mock_plugin)
    return engine

@pytest.mark.asyncio
async def test_thalamus_pre_process(context_engine, mock_plugin):
    thalamus = Thalamus(context_engine=context_engine)

    result = await thalamus.pre_process("hello")
    assert result == "pre_hello"
    mock_plugin.pre_process.assert_called_once_with("hello")

@pytest.mark.asyncio
async def test_working_memory_compress(context_engine, mock_plugin):
    wm = WorkingMemory(limit=2, context_engine=context_engine)
    pad = PADState(0,0,0)

    e1 = MemoryEntry("1", 0.5, pad)
    e2 = MemoryEntry("2", 0.5, pad)
    e3 = MemoryEntry("3", 0.5, pad)

    # Adding below limit should do nothing to compress
    await wm.add(e1)
    await wm.add(e2)
    assert len(wm.get_entries()) == 2
    mock_plugin.compress.assert_not_called()

    # Adding 3rd should trigger compress of the oldest 2 (e1, e2)
    # The mock plugin compress takes the list and returns a list of just the first item (e1)
    # The fallback/insert logic in working memory expects a single entry back, not a list, but handles whatever is returned.
    # We will adjust the mock to just return a single entry representing the compressed item.
    mock_plugin.compress = AsyncMock(side_effect=lambda x: x[0])
    await wm.add(e3)

    mock_plugin.compress.assert_called_once()
    entries = wm.get_entries()
    assert len(entries) == 2  # The 1 compressed + the new 3rd
    assert entries[0] == e1   # The result of our mock compression
    assert entries[1] == e3

@pytest.mark.asyncio
async def test_working_memory_post_process(context_engine, mock_plugin):
    wm = WorkingMemory(limit=2, context_engine=context_engine)
    pad = PADState(0,0,0)

    e1 = MemoryEntry("1", 0.5, pad)
    await wm.add(e1)

    result = await wm.get_entries_async()
    mock_plugin.post_process.assert_called_once()
    assert result[0] == f"post_{e1}"
