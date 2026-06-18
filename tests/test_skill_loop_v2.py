import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.learning.skill_loop_v2 import HermesSkillCreationLoopV2
from magda_agent.memory.procedural import ProceduralMemory
from magda_agent.llm_client import LLMClient

@pytest.fixture
def procedural_memory() -> ProceduralMemory:
    mem = MagicMock(spec=ProceduralMemory)
    return mem

@pytest.fixture
def llm_client() -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.chat_completion = AsyncMock()
    return llm

@pytest.mark.asyncio
async def test_analyze_and_create_skill_success(procedural_memory: ProceduralMemory, llm_client: LLMClient) -> None:
    llm_client.chat_completion.return_value = "def fetch_data(url):\n    pass"
    loop = HermesSkillCreationLoopV2(procedural_memory, llm_client)

    trace = [{"action": "get url", "outcome": "data fetched"}]

    result = await loop.analyze_and_create_skill("fetch data from api", trace, user_id=1)

    assert result == "fetch_data"
    procedural_memory.store_procedure.assert_called_once_with(
        name="fetch_data",
        procedure="def fetch_data(url):\n    pass",
        user_id=1,
        metadata={"source_task": "fetch data from api", "type": "python_skill_loop_v2"}
    )

@pytest.mark.asyncio
async def test_analyze_and_create_skill_not_reusable(procedural_memory: ProceduralMemory, llm_client: LLMClient) -> None:
    llm_client.chat_completion.return_value = "NOT_REUSABLE"
    loop = HermesSkillCreationLoopV2(procedural_memory, llm_client)

    trace = [{"action": "click button", "outcome": "clicked"}]

    result = await loop.analyze_and_create_skill("click", trace)

    assert result is None
    procedural_memory.store_procedure.assert_not_called()
