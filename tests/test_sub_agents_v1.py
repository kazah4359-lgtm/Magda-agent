import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.architecture.sub_agents import SubAgentRPCManager
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock(return_value="Task executed successfully.")
    return llm

@pytest.fixture
def manager(mock_llm):
    return SubAgentRPCManager(llm=mock_llm)

@pytest.mark.asyncio
async def test_spawn_and_execute_task(manager, mock_llm):
    # Mock GitWorktreeManager to avoid actual git operations in tests
    with pytest.MonkeyPatch.context() as m:
        mock_worktree_manager = MagicMock()
        mock_worktree_manager.create_worktree_async = AsyncMock(return_value="/tmp/worktree1")
        mock_worktree_manager.remove_worktree_async = AsyncMock()
        m.setattr("magda_agent.agents.sub_agent.GitWorktreeManager", MagicMock(return_value=mock_worktree_manager))

        await manager.spawn_agent("agent_1", context={"role": "developer"})

        assert "agent_1" in manager.active_agents
        agent = manager.active_agents["agent_1"]
        assert "developer" in agent.system_prompt
        assert agent.use_isolation is True

        result = await manager.execute_task_rpc("agent_1", "Write a test")

        assert result["status"] == "success"
        assert result["agent_id"] == "agent_1"
        assert result["result"] == "Task executed successfully."
        mock_llm.chat_completion.assert_called_once()

        await manager.kill_agent("agent_1")
        assert "agent_1" not in manager.active_agents

@pytest.mark.asyncio
async def test_execute_task_agent_not_found(manager):
    result = await manager.execute_task_rpc("non_existent", "Task")
    assert result["status"] == "error"
    assert "not found" in result["error"]
