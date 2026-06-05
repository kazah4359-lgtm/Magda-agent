import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.emotions.style_adapter import StyleAdapter
from magda_agent.emotions.engine import PADState, EmotionalEngine
from magda_agent.consciousness.core import Consciousness

def test_style_adapter_high_arousal():
    adapter = StyleAdapter()
    pad = PADState(pleasure=0.0, arousal=0.8, dominance=0.0)
    prompt = adapter.get_style_prompt(pad)

    assert "State is high arousal" in prompt
    assert "concise" in prompt

def test_style_adapter_low_arousal():
    adapter = StyleAdapter()
    pad = PADState(pleasure=0.0, arousal=-0.8, dominance=0.0)
    prompt = adapter.get_style_prompt(pad)

    assert "State is low arousal" in prompt
    assert "relaxed" in prompt

def test_style_adapter_with_user_model_technical():
    adapter = StyleAdapter()
    pad = PADState(0, 0, 0)
    user_model = {"expertise_level": "technical"}
    prompt = adapter.get_style_prompt(pad, user_model)

    assert "highly technical" in prompt
    assert "code examples" in prompt

def test_style_adapter_with_user_model_beginner_and_short():
    adapter = StyleAdapter()
    pad = PADState(0, 0, 0)
    user_model = {
        "expertise_level": "beginner",
        "preferences": {"short_answers": True}
    }
    prompt = adapter.get_style_prompt(pad, user_model)

    assert "beginner" in prompt
    assert "simply" in prompt
    assert "very short, bulleted answers" in prompt

def test_style_adapter_combined():
    adapter = StyleAdapter()
    pad = PADState(pleasure=0.8, arousal=0.8, dominance=0.8)
    user_model = {"communication_style": "informal"}
    prompt = adapter.get_style_prompt(pad, user_model)

    assert "high arousal" in prompt
    assert "high pleasure" in prompt
    assert "high dominance" in prompt
    assert "informal" in prompt

@pytest.mark.asyncio
async def test_consciousness_integration_with_style_adapter():
    mock_llm = MagicMock()
    mock_llm.get_system_prompt.return_value = "Base system prompt"
    mock_llm.chat_completion = AsyncMock(return_value="Mock response")

    mock_emotions = MagicMock(spec=EmotionalEngine)
    mock_emotions.get_state_history.return_value = (PADState(arousal=0.9), [])

    mock_memory = MagicMock()
    mock_memory.add_memory = AsyncMock()
    mock_memory.retrieve_relevant.return_value = []
    mock_memory.working_memory.get_entries.return_value = []

    mock_skills = MagicMock()

    mock_user_model = MagicMock()
    mock_user_model.get_model.return_value = {"expertise_level": "technical"}

    adapter = StyleAdapter()

    consciousness = Consciousness(
        llm=mock_llm,
        emotions=mock_emotions,
        memory=mock_memory,
        skills=mock_skills,
        style_adapter=adapter,
        user_model=mock_user_model
    )

    await consciousness.process_input("Hello", user_id=123)

    # Check that LLM chat_completion was called with the modified system prompt
    call_args = mock_llm.chat_completion.call_args[0][0]
    system_prompt_used = call_args[0]["content"]

    assert "Base system prompt" in system_prompt_used
    assert "State is high arousal" in system_prompt_used
    assert "highly technical" in system_prompt_used

@pytest.mark.asyncio
async def test_consciousness_integration_no_style_adapter():
    mock_llm = MagicMock()
    mock_llm.get_system_prompt.return_value = "Base system prompt"
    mock_llm.chat_completion = AsyncMock(return_value="Mock response")

    mock_emotions = MagicMock(spec=EmotionalEngine)
    mock_memory = MagicMock()
    mock_memory.add_memory = AsyncMock()
    mock_memory.retrieve_relevant.return_value = []
    mock_memory.working_memory.get_entries.return_value = []

    mock_skills = MagicMock()

    # Init without style adapter
    consciousness = Consciousness(
        llm=mock_llm,
        emotions=mock_emotions,
        memory=mock_memory,
        skills=mock_skills
    )

    await consciousness.process_input("Hello")

    call_args = mock_llm.chat_completion.call_args[0][0]
    system_prompt_used = call_args[0]["content"]

    assert system_prompt_used == "Base system prompt"
