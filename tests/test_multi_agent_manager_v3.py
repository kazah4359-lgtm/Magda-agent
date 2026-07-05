import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from magda_agent.llm_client import LLMClient
from magda_agent.agents.multi_agent_manager_v3 import MultiAgentManagerV3
from magda_agent.agents.sub_agent import SubAgent

@pytest.fixture
def mock_llm_client():
    llm = LLMClient(api_key="test-key")
    llm.chat_completion = AsyncMock()
    llm.chat_completion.return_value = "Mocked subagent response"
    return llm

@pytest.fixture
def mock_subagent_compressor():
    with patch("magda_agent.agents.sub_agent.SubagentContextCompressor") as mock:
        compressor_instance = mock.return_value
        compressor_instance.compress_context = AsyncMock(side_effect=lambda ctx: ctx)
        yield compressor_instance

def test_spawn_subagent(mock_llm_client, mock_subagent_compressor):
    manager = MultiAgentManagerV3(llm=mock_llm_client)
    subagent = manager.spawn_subagent(system_prompt="Test Prompt", use_isolation=False)

    assert isinstance(subagent, SubAgent)
    assert subagent.llm == mock_llm_client
    assert subagent.system_prompt == "Test Prompt"
    assert subagent.use_isolation is False

@pytest.mark.asyncio
async def test_run_parallel_tasks(mock_llm_client, mock_subagent_compressor):
    manager = MultiAgentManagerV3(llm=mock_llm_client)
    tasks = [
        {"task": "Task 1", "system_prompt": "Prompt 1"},
        {"task": "Task 2", "system_prompt": "Prompt 2"}
    ]
    context = "Shared Context"

    # Mocking chat_completion on LLMClient
    mock_llm_client.chat_completion.side_effect = ["Response 1", "Response 2"]

    results = await manager.run_parallel_tasks(tasks, context, use_isolation=False)

    assert len(results) == 2
    assert "Response" in results[0]
    assert "Response" in results[1]

    assert mock_llm_client.chat_completion.call_count == 2
