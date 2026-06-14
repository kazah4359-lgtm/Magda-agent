import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.learning.skill_creator import SkillCreator
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
async def test_analyze_chat_and_generate_skill_success(procedural_memory: ProceduralMemory, llm_client: LLMClient) -> None:
    llm_client.chat_completion.return_value = "def new_dynamic_skill(param):\n    return param"
    creator = SkillCreator(procedural_memory, llm_client)
    chat_history = [{"role": "user", "content": "do X"}, {"role": "assistant", "content": "did X"}]

    await creator.analyze_chat_and_generate_skill(chat_history, user_id=123)

    procedural_memory.store_procedure.assert_called_once_with(
        name="new_dynamic_skill",
        procedure="def new_dynamic_skill(param):\n    return param",
        user_id=123,
        metadata={"source": "dynamic_generation", "type": "python_code"}
    )

@pytest.mark.asyncio
async def test_analyze_chat_and_generate_skill_no_pattern(procedural_memory: ProceduralMemory, llm_client: LLMClient) -> None:
    llm_client.chat_completion.return_value = "NO_PATTERN"
    creator = SkillCreator(procedural_memory, llm_client)
    chat_history = [{"role": "user", "content": "hello"}]

    await creator.analyze_chat_and_generate_skill(chat_history)

    procedural_memory.store_procedure.assert_not_called()
