import pytest
import os
import json
import asyncio
import time
from unittest.mock import AsyncMock
from magda_agent.memory.hermes_persistent import HermesPersistentMemory

@pytest.mark.asyncio
async def test_hermes_initial_profile(tmp_path):
    user_id = 111
    memory = HermesPersistentMemory(persist_dir=str(tmp_path))
    profile = memory.get_profile(user_id)

    assert profile["user_id"] == user_id
    assert profile["traits"] == []
    assert profile["facts"] == []
    assert profile["stats"]["interaction_count"] == 0
    assert "first_seen" in profile["stats"]

@pytest.mark.asyncio
async def test_hermes_update_interaction(tmp_path):
    user_id = 222
    mock_llm = AsyncMock()
    mock_llm.chat_completion.return_value = '{"traits": ["helpful"], "facts": ["Lives in London"]}'

    memory = HermesPersistentMemory(persist_dir=str(tmp_path), llm=mock_llm)

    await memory.update_from_interaction(user_id, "Hello, I live in London and I try to be helpful.")

    profile = memory.get_profile(user_id)
    assert profile["stats"]["interaction_count"] == 1
    assert "helpful" in profile["traits"]
    assert "Lives in London" in profile["facts"]

    # Second interaction
    mock_llm.chat_completion.return_value = '{"traits": ["technical"], "facts": ["Uses Python"]}'
    await memory.update_from_interaction(user_id, "I also use Python for my technical projects.")

    profile = memory.get_profile(user_id)
    assert profile["stats"]["interaction_count"] == 2
    assert "helpful" in profile["traits"]
    assert "technical" in profile["traits"]
    assert "Lives in London" in profile["facts"]
    assert "Uses Python" in profile["facts"]

@pytest.mark.asyncio
async def test_hermes_persistence(tmp_path):
    user_id = 333
    memory = HermesPersistentMemory(persist_dir=str(tmp_path))
    profile = memory.get_profile(user_id)
    profile["traits"].append("persistent")
    memory.save_profile(user_id, profile)

    # Re-initialize
    memory2 = HermesPersistentMemory(persist_dir=str(tmp_path))
    profile2 = memory2.get_profile(user_id)
    assert "persistent" in profile2["traits"]
    assert profile2["user_id"] == user_id
