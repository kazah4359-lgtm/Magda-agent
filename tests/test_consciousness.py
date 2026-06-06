import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.modules.setdefault("transformers", MagicMock())

from magda_agent.consciousness.core import Consciousness
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills.registry import SkillRegistry
from magda_agent.safety.policy import PolicyLayer
from magda_agent.llm_client import LLMClient


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=LLMClient)
    llm.get_system_prompt.return_value = "System prompt"
    llm.chat_completion = AsyncMock(return_value="Consciousness response")
    return llm


@pytest.mark.asyncio
async def test_process_input_returns_llm_response(mock_llm: MagicMock, tmp_path) -> None:
    memory = MemorySystem(
        persist_directory=str(tmp_path / "episodic"),
        llm=mock_llm,
        short_term_limit=5,
    )
    consciousness = Consciousness(
        llm=mock_llm,
        emotions=EmotionalEngine(),
        memory=memory,
        skills=SkillRegistry(policy_layer=PolicyLayer()),
    )

    response = await consciousness.process_input("Hello", user_id=42)

    assert response == "Consciousness response"
    assert mock_llm.chat_completion.await_count >= 1


@pytest.mark.asyncio
async def test_brainstem_reflex_bypasses_llm(mock_llm: MagicMock, tmp_path) -> None:
    from magda_agent.reflexes.brainstem import Brainstem

    memory = MemorySystem(
        persist_directory=str(tmp_path / "episodic"),
        llm=mock_llm,
        short_term_limit=5,
    )
    consciousness = Consciousness(
        llm=mock_llm,
        emotions=EmotionalEngine(),
        memory=memory,
        skills=SkillRegistry(),
        brainstem=Brainstem(),
    )

    response = await consciousness.process_input("stop", user_id=1)

    assert "halted" in response.lower() or "stop" in response.lower()
    mock_llm.chat_completion.assert_not_awaited()
