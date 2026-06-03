import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.emotions.attachment import AttachmentModel
from magda_agent.consciousness.core import Consciousness
from magda_agent.emotions.engine import EmotionalEngine
from magda_agent.memory.storage import MemorySystem
from magda_agent.skills.registry import SkillRegistry

def test_attachment_progression():
    model = AttachmentModel()
    user_id = 123

    # Initial state (0 interactions)
    assert model.get_level(user_id) == "stranger"
    assert "Stranger" in model.get_attachment_prompt(user_id)

    # 1-2 interactions -> stranger
    model.record_interaction(user_id)
    model.record_interaction(user_id)
    assert model.get_level(user_id) == "stranger"

    # 3-5 interactions -> acquaintance
    model.record_interaction(user_id)
    assert model.get_level(user_id) == "acquaintance"
    assert "Acquaintance" in model.get_attachment_prompt(user_id)

    model.record_interaction(user_id)
    model.record_interaction(user_id)
    assert model.get_level(user_id) == "acquaintance"

    # 6-9 interactions -> friend
    model.record_interaction(user_id)
    assert model.get_level(user_id) == "friend"
    assert "Friend" in model.get_attachment_prompt(user_id)

    model.record_interaction(user_id)
    model.record_interaction(user_id)
    model.record_interaction(user_id)
    assert model.get_level(user_id) == "friend"

    # 10+ interactions -> close_friend
    model.record_interaction(user_id)
    assert model.get_level(user_id) == "close_friend"
    assert "Close Friend" in model.get_attachment_prompt(user_id)

@pytest.mark.asyncio
async def test_consciousness_attachment_integration():
    llm_mock = MagicMock()
    llm_mock.get_system_prompt.return_value = "Base System Prompt"
    llm_mock.chat_completion = AsyncMock(return_value="Mocked response")

    emotions = EmotionalEngine()
    emotions.get_summary = MagicMock(return_value="Current Emotion: Neutral")

    memory = MemorySystem()
    skills = SkillRegistry()

    attachment = AttachmentModel()
    user_id = 999

    consciousness = Consciousness(
        llm=llm_mock,
        emotions=emotions,
        memory=memory,
        skills=skills,
        attachment=attachment
    )

    # First interaction - should be stranger
    await consciousness.process_input("Hello", user_id)

    # Check that record_interaction was called implicitly because level should have increased by 1
    assert attachment.user_interactions[user_id] == 1

    # Check get_system_prompt was called with the attachment prompt
    call_args = llm_mock.get_system_prompt.call_args[1]
    assert "Stranger" in call_args["emotions"]
    assert "Current Emotion: Neutral" in call_args["emotions"]

    # Fast forward to 10 interactions (close_friend)
    attachment.user_interactions[user_id] = 9
    await consciousness.process_input("We know each other well", user_id)

    # Now it should be 10 (close friend)
    assert attachment.user_interactions[user_id] == 10
    call_args_2 = llm_mock.get_system_prompt.call_args[1]
    assert "Close Friend" in call_args_2["emotions"]
