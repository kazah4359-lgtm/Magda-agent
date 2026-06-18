import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from magda_agent.agents.hierarchical_delegation_v2 import HierarchicalDelegatorV2
from magda_agent.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.chat_completion = AsyncMock(return_value="Mocked SubAgent response")
    return client

@pytest.fixture
def mock_worktree_manager():
    manager = MagicMock()
    manager.create_worktree_async = AsyncMock(return_value="/tmp/mock_worktree")
    manager.remove_worktree_async = AsyncMock()
    return manager

@pytest.mark.asyncio
@patch('magda_agent.agents.hierarchical_delegation_v2.GitWorktreeManager')
async def test_delegate_single_task(mock_gwm_class, mock_llm_client, mock_worktree_manager):
    mock_gwm_class.return_value = mock_worktree_manager
    delegator = HierarchicalDelegatorV2(llm=mock_llm_client)

    task = {"description": "Single leaf task"}

    # We must patch SubAgent execution to avoid actual context compression logic that might call llm unpredictably.
    with patch('magda_agent.agents.hierarchical_delegation_v2.SubAgent.execute', new_callable=AsyncMock) as mock_sub_execute:
        mock_sub_execute.return_value = "Leaf Task Result"
        result = await delegator.delegate_task(task, context="Initial Context")

        assert result == "Leaf Task Result"
        mock_worktree_manager.create_worktree_async.assert_awaited_once()
        mock_worktree_manager.remove_worktree_async.assert_awaited_once_with("/tmp/mock_worktree")

@pytest.mark.asyncio
@patch('magda_agent.agents.hierarchical_delegation_v2.GitWorktreeManager')
async def test_delegate_recursive_tasks(mock_gwm_class, mock_llm_client, mock_worktree_manager):
    mock_gwm_class.return_value = mock_worktree_manager
    delegator = HierarchicalDelegatorV2(llm=mock_llm_client)

    task = {
        "description": "Parent task",
        "sub_tasks": [
            {"description": "Child 1"},
            {"description": "Child 2"}
        ]
    }

    with patch('magda_agent.agents.hierarchical_delegation_v2.SubAgent.execute', new_callable=AsyncMock) as mock_sub_execute:
        # Mock subagent returning distinct results per call
        mock_sub_execute.side_effect = ["Result Child 1", "Result Child 2", "Parent Final Result"]

        result = await delegator.delegate_task(task, context="Initial Context")

        assert result == "Parent Final Result"

        # 2 children + 1 parent = 3 worktree creations
        assert mock_worktree_manager.create_worktree_async.await_count == 3
        assert mock_worktree_manager.remove_worktree_async.await_count == 3

        # Check that parent received combined context
        parent_call_kwargs = mock_sub_execute.call_args_list[-1].kwargs
        parent_context = parent_call_kwargs['context']
        assert "Sub-task Result 1:\nResult Child 1" in parent_context
        assert "Sub-task Result 2:\nResult Child 2" in parent_context

@pytest.mark.asyncio
@patch('magda_agent.agents.hierarchical_delegation_v2.GitWorktreeManager')
async def test_max_depth_reached(mock_gwm_class, mock_llm_client, mock_worktree_manager):
    mock_gwm_class.return_value = mock_worktree_manager
    delegator = HierarchicalDelegatorV2(llm=mock_llm_client)

    task = {
        "description": "Deeply nested task",
        "sub_tasks": [
            {"description": "Child 1"}
        ]
    }

    with patch('magda_agent.agents.hierarchical_delegation_v2.SubAgent.execute', new_callable=AsyncMock) as mock_sub_execute:
        mock_sub_execute.return_value = "Forced Execution Result"

        # Call with depth = max_depth to trigger early return
        result = await delegator.delegate_task(task, context="Initial Context", depth=3, max_depth=3)

        assert result == "Forced Execution Result"
        # It executed leaf directly, ignoring sub_tasks
        mock_worktree_manager.create_worktree_async.assert_awaited_once()
