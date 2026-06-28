import pytest
from unittest.mock import MagicMock
from magda_agent.memory.compression_hook import CompressionHookPlugin

@pytest.mark.asyncio
async def test_compression_hook_plugin_bootstrap():
    plugin = CompressionHookPlugin()
    mock_memory_system = MagicMock()
    config = {"memory_system": mock_memory_system}

    await plugin.bootstrap(config)
    assert plugin.memory_system == mock_memory_system

def test_compression_hook_plugin_before_retrieval_compress():
    mock_memory_system = MagicMock()
    mock_working_memory = MagicMock()

    # Set limit
    mock_working_memory.limit = 5

    # Create 6 entries to trigger compression (6 >= 5, so it should remove 6 - 5 + 1 = 2 entries)
    entry1 = MagicMock(id="id1", importance=0.9)
    entry2 = MagicMock(id="id2", importance=0.1) # should be removed first
    entry3 = MagicMock(id="id3", importance=0.5)
    entry4 = MagicMock(id="id4", importance=0.8)
    entry5 = MagicMock(id="id5", importance=0.2) # should be removed second
    entry6 = MagicMock(id="id6", importance=0.6)

    mock_working_memory.get_entries.return_value = [entry1, entry2, entry3, entry4, entry5, entry6]
    mock_memory_system.working_memory = mock_working_memory

    plugin = CompressionHookPlugin(memory_system=mock_memory_system)

    query = "test query"
    user_id = 1

    result_query = plugin.before_retrieval(query, user_id)

    # query should be unchanged
    assert result_query == query

    # It should have called get_entries
    mock_working_memory.get_entries.assert_called_once_with(user_id)

    # It should have removed 2 entries: id2 and id5 (the ones with lowest importance: 0.1 and 0.2)
    assert mock_working_memory.remove.call_count == 2
    mock_working_memory.remove.assert_any_call("id2", user_id)
    mock_working_memory.remove.assert_any_call("id5", user_id)

def test_compression_hook_plugin_before_retrieval_no_compress():
    mock_memory_system = MagicMock()
    mock_working_memory = MagicMock()

    # Set limit
    mock_working_memory.limit = 5

    # Create 4 entries (no compression needed)
    entry1 = MagicMock(id="id1", importance=0.9)
    entry2 = MagicMock(id="id2", importance=0.1)

    mock_working_memory.get_entries.return_value = [entry1, entry2]
    mock_memory_system.working_memory = mock_working_memory

    plugin = CompressionHookPlugin(memory_system=mock_memory_system)

    plugin.before_retrieval("query", 1)

    # Should not remove anything
    mock_working_memory.remove.assert_not_called()
