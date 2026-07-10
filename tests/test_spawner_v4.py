import pytest
from unittest.mock import AsyncMock, patch
from magda_agent.agents.spawner_v4 import SubagentSpawnerV4
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_subagent_spawner_v4_success() -> None:
    """Test successful task execution, compression, and isolation for multiple tasks."""
    mock_llm = AsyncMock(spec=LLMClient)
    # The first call to LLM could be from the context compressor (if context is long enough)
    # or just chat_completion. Let's mock the compressor directly for predictable behavior.

    with patch("magda_agent.agents.spawner_v4.AgentWorktreeIsolationV3") as mock_isolation_class, \
         patch("magda_agent.agents.spawner_v4.SubagentContextCompressor") as mock_compressor_class:

        mock_isolation_instance = mock_isolation_class.return_value
        mock_isolation_instance.setup_isolation = AsyncMock(side_effect=["/tmp/wt1", "/tmp/wt2"])
        mock_isolation_instance.teardown_isolation = AsyncMock()

        mock_compressor_instance = mock_compressor_class.return_value
        mock_compressor_instance.compress_context = AsyncMock(return_value="Compressed Context")

        mock_llm.chat_completion.side_effect = ["Result 1", "Result 2"]

        spawner = SubagentSpawnerV4(llm=mock_llm)
        tasks = [
            {"description": "Task 1", "system_prompt": "Sys 1"},
            {"description": "Task 2", "system_prompt": "Sys 2"}
        ]

        results = await spawner.spawn_and_execute(tasks=tasks, base_context="Very long shared context " * 100)

        assert len(results) == 2
        assert "Result 1" in results
        assert "Result 2" in results

        assert mock_compressor_instance.compress_context.call_count == 2
        assert mock_isolation_instance.setup_isolation.call_count == 2
        assert mock_isolation_instance.teardown_isolation.call_count == 2
        assert mock_llm.chat_completion.call_count == 2

@pytest.mark.asyncio
async def test_subagent_spawner_v4_compression_failure_handled() -> None:
    """Test behavior when compression fails (though our compressor mock handles it, let's test if it raises)."""
    mock_llm = AsyncMock(spec=LLMClient)

    with patch("magda_agent.agents.spawner_v4.AgentWorktreeIsolationV3") as mock_isolation_class, \
         patch("magda_agent.agents.spawner_v4.SubagentContextCompressor") as mock_compressor_class:

        mock_isolation_instance = mock_isolation_class.return_value
        mock_isolation_instance.setup_isolation = AsyncMock()
        mock_isolation_instance.teardown_isolation = AsyncMock()

        mock_compressor_instance = mock_compressor_class.return_value
        mock_compressor_instance.compress_context = AsyncMock(side_effect=Exception("Compression Error"))

        spawner = SubagentSpawnerV4(llm=mock_llm)
        tasks = [{"description": "Task 1"}]

        results = await spawner.spawn_and_execute(tasks=tasks, base_context="Shared Context")

        assert len(results) == 1
        assert "Error executing SubAgent task: Compression Error" in results[0]

        # Teardown should still be called even if setup didn't happen because it's in finally
        # Actually, in our code, if compress fails, teardown is called but setup_isolation wasn't.
        # teardown_isolation expects an agent_id. It should be handled gracefully.
        assert mock_isolation_instance.teardown_isolation.call_count == 1
        assert mock_isolation_instance.setup_isolation.call_count == 0

@pytest.mark.asyncio
async def test_subagent_spawner_v4_setup_failure() -> None:
    """Test behavior when isolation setup fails."""
    mock_llm = AsyncMock(spec=LLMClient)

    with patch("magda_agent.agents.spawner_v4.AgentWorktreeIsolationV3") as mock_isolation_class, \
         patch("magda_agent.agents.spawner_v4.SubagentContextCompressor") as mock_compressor_class:

        mock_isolation_instance = mock_isolation_class.return_value
        mock_isolation_instance.setup_isolation = AsyncMock(side_effect=Exception("Git Error"))
        mock_isolation_instance.teardown_isolation = AsyncMock()

        mock_compressor_instance = mock_compressor_class.return_value
        mock_compressor_instance.compress_context = AsyncMock(return_value="CompCtx")

        spawner = SubagentSpawnerV4(llm=mock_llm)
        tasks = [{"description": "Task 1"}]

        results = await spawner.spawn_and_execute(tasks=tasks, base_context="Shared")

        assert len(results) == 1
        assert "Error executing SubAgent task: Git Error" in results[0]

        assert mock_compressor_instance.compress_context.call_count == 1
        assert mock_isolation_instance.setup_isolation.call_count == 1
        assert mock_isolation_instance.teardown_isolation.call_count == 1
        assert mock_llm.chat_completion.call_count == 0

@pytest.mark.asyncio
async def test_subagent_spawner_v4_execution_failure() -> None:
    """Test behavior when LLM execution fails, ensuring teardown still occurs."""
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat_completion.side_effect = Exception("LLM Error")

    with patch("magda_agent.agents.spawner_v4.AgentWorktreeIsolationV3") as mock_isolation_class, \
         patch("magda_agent.agents.spawner_v4.SubagentContextCompressor") as mock_compressor_class:

        mock_isolation_instance = mock_isolation_class.return_value
        mock_isolation_instance.setup_isolation = AsyncMock(return_value="/tmp/wt1")
        mock_isolation_instance.teardown_isolation = AsyncMock()

        mock_compressor_instance = mock_compressor_class.return_value
        mock_compressor_instance.compress_context = AsyncMock(return_value="CompCtx")

        spawner = SubagentSpawnerV4(llm=mock_llm)
        tasks = [{"description": "Task 1"}]

        results = await spawner.spawn_and_execute(tasks=tasks, base_context="Shared")

        assert len(results) == 1
        assert "Error executing SubAgent task: LLM Error" in results[0]

        assert mock_compressor_instance.compress_context.call_count == 1
        assert mock_isolation_instance.setup_isolation.call_count == 1
        assert mock_llm.chat_completion.call_count == 1
        assert mock_isolation_instance.teardown_isolation.call_count == 1
