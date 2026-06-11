import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from magda_agent.architecture.hierarchical_delegation import HierarchicalDelegator
from magda_agent.llm_client import LLMClient

@pytest.mark.asyncio
async def test_delegate_tasks_success():
    """Test that HierarchicalDelegator successfully delegates tasks to subagents in isolation."""
    llm = MagicMock(spec=LLMClient)

    tasks = ["Task 1", "Task 2"]
    base_context = "Initial context"

    mock_sub_agent_instance = MagicMock()
    mock_sub_agent_instance.execute = AsyncMock(side_effect=["Result 1", "Result 2"])

    with patch("magda_agent.architecture.hierarchical_delegation.SubAgent", return_value=mock_sub_agent_instance) as mock_sub_agent_cls:
        delegator = HierarchicalDelegator(llm=llm)
        results = await delegator.delegate_tasks(tasks, base_context)

        assert len(results) == 2
        assert results[0] == {"task": "Task 1", "status": "success", "result": "Result 1"}
        assert results[1] == {"task": "Task 2", "status": "success", "result": "Result 2"}

        # SubAgent should be instantiated twice with use_isolation=True
        assert mock_sub_agent_cls.call_count == 2
        mock_sub_agent_cls.assert_any_call(llm=llm, use_isolation=True)

        # Execute should be called twice
        assert mock_sub_agent_instance.execute.call_count == 2
        mock_sub_agent_instance.execute.assert_any_call(task="Task 1", context=base_context)
        mock_sub_agent_instance.execute.assert_any_call(task="Task 2", context=base_context)

@pytest.mark.asyncio
async def test_delegate_tasks_failure():
    """Test that HierarchicalDelegator handles execution errors."""
    llm = MagicMock(spec=LLMClient)

    tasks = ["Task 1"]

    mock_sub_agent_instance = MagicMock()
    mock_sub_agent_instance.execute = AsyncMock(side_effect=Exception("Failed"))

    with patch("magda_agent.architecture.hierarchical_delegation.SubAgent", return_value=mock_sub_agent_instance):
        delegator = HierarchicalDelegator(llm=llm)
        results = await delegator.delegate_tasks(tasks, "")

        assert len(results) == 1
        assert results[0] == {"task": "Task 1", "status": "error", "error": "Failed"}
