import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.reflexes.brainstem import Brainstem
from magda_agent.consciousness.core import Consciousness

def test_brainstem_reflex_stop():
    brainstem = Brainstem()
    response = brainstem.process_reflex("stop")
    assert response == "Emergency Stop triggered. Halting current processes."

    response_ru = brainstem.process_reflex(" СТОП ")
    assert response_ru == "Emergency Stop triggered. Halting current processes."

def test_brainstem_reflex_help():
    brainstem = Brainstem()
    response = brainstem.process_reflex("help")
    assert response == "Emergency Assistance reflex triggered. How can I help you immediately?"

    response_ru = brainstem.process_reflex("помощь")
    assert response_ru == "Emergency Assistance reflex triggered. How can I help you immediately?"

def test_brainstem_no_reflex():
    brainstem = Brainstem()
    response = brainstem.process_reflex("hello how are you")
    assert response is None

    response_empty = brainstem.process_reflex("")
    assert response_empty is None

@pytest.mark.asyncio
async def test_brainstem_integration():
    # Setup mocks
    mock_llm = MagicMock()
    mock_emotions = MagicMock()
    mock_memory = MagicMock()
    mock_skills = MagicMock()

    brainstem = Brainstem()

    core = Consciousness(
        llm=mock_llm,
        emotions=mock_emotions,
        memory=mock_memory,
        skills=mock_skills,
        brainstem=brainstem
    )

    # Test triggering reflex
    response = await core.process_input("stop")

    # Verify we got the early return reflex response
    assert response == "Emergency Stop triggered. Halting current processes."

    # Verify core did NOT process further
    mock_emotions.update.assert_not_called()
    mock_llm.chat_completion.assert_not_called()

@pytest.mark.asyncio
async def test_brainstem_integration_no_reflex():
    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value="mock response")
    mock_emotions = MagicMock()
    mock_memory = MagicMock()
    mock_memory.add_memory = AsyncMock()
    mock_skills = MagicMock()

    brainstem = Brainstem()

    core = Consciousness(
        llm=mock_llm,
        emotions=mock_emotions,
        memory=mock_memory,
        skills=mock_skills,
        brainstem=brainstem
    )

    # Test no reflex
    response = await core.process_input("hello")

    # Verify core processed normally
    assert response == "mock response"
    mock_emotions.update.assert_called()
    mock_llm.chat_completion.assert_called_once()
