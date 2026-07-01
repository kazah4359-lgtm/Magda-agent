import pytest
import os
import json
import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock
from magda_agent.memory.hermes_persistent_v2 import HermesPersistentMemoryV2

@pytest.mark.asyncio
async def test_hermes_v2_initial_profile(tmp_path: Path) -> None:
    """
    Tests that a fresh user ID returns a valid initial profile with default empty fields.
    """
    user_id = 111
    memory = HermesPersistentMemoryV2(persist_dir=str(tmp_path))
    profile = memory.get_profile(user_id)

    assert profile["user_id"] == user_id
    assert profile["traits"] == []
    assert profile["facts"] == []
    assert profile["interests"] == []
    assert profile["communication_style"] == []
    assert profile["goals"] == []
    assert profile["stats"]["interaction_count"] == 0
    assert "first_seen" in profile["stats"]
    assert profile["version"] == 2

@pytest.mark.asyncio
async def test_hermes_v2_update_interaction(tmp_path: Path) -> None:
    """
    Tests that interactions are properly sent to the mocked LLM, and that the returned traits/facts/etc.
    are merged correctly into the user's persistent profile.
    """
    user_id = 222
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = '{"traits": ["helpful"], "facts": ["Lives in London"], "interests": ["AI"], "communication_style": ["formal"], "goals": ["Learn Python"]}'

    memory = HermesPersistentMemoryV2(persist_dir=str(tmp_path), llm=mock_llm)

    await memory.update_from_interaction(user_id, "Hello, I live in London and I try to be helpful. I am interested in AI and my goal is to learn Python.")

    profile = memory.get_profile(user_id)
    assert profile["stats"]["interaction_count"] == 1
    assert "helpful" in profile["traits"]
    assert "Lives in London" in profile["facts"]
    assert "AI" in profile["interests"]
    assert "formal" in profile["communication_style"]
    assert "Learn Python" in profile["goals"]

    # Second interaction
    mock_llm.chat_completion.return_value = '{"traits": ["technical"], "facts": ["Uses Python"], "interests": ["Robotics"], "communication_style": ["direct"], "goals": ["Build a robot"]}'
    await memory.update_from_interaction(user_id, "I also use Python for my technical projects. I like robotics and want to build a robot.")

    profile = memory.get_profile(user_id)
    assert profile["stats"]["interaction_count"] == 2
    assert "helpful" in profile["traits"]
    assert "technical" in profile["traits"]
    assert "Lives in London" in profile["facts"]
    assert "Uses Python" in profile["facts"]
    assert "AI" in profile["interests"]
    assert "Robotics" in profile["interests"]
    assert "formal" in profile["communication_style"]
    assert "direct" in profile["communication_style"]
    assert "Learn Python" in profile["goals"]
    assert "Build a robot" in profile["goals"]

@pytest.mark.asyncio
async def test_hermes_v2_persistence(tmp_path: Path) -> None:
    """
    Tests that modifications saved to disk can be accurately loaded back by a new memory instance.
    """
    user_id = 333
    memory = HermesPersistentMemoryV2(persist_dir=str(tmp_path))
    profile = memory.get_profile(user_id)
    profile["traits"].append("persistent")
    profile["interests"].append("coding")
    memory.save_profile(user_id, profile)

    # Re-initialize
    memory2 = HermesPersistentMemoryV2(persist_dir=str(tmp_path))
    profile2 = memory2.get_profile(user_id)
    assert "persistent" in profile2["traits"]
    assert "coding" in profile2["interests"]
    assert profile2["user_id"] == user_id
    assert profile2["version"] == 2
