import pytest
import os
import json
import asyncio
from unittest.mock import AsyncMock, patch
from magda_agent.user_model.model import UserModel

def test_user_model_persistence(tmp_path):
    user_id = 123
    model = UserModel(persist_dir=str(tmp_path))

    # Check default model
    data = model.get_model(user_id)
    assert data["expertise_level"] == "unknown"

def test_update_user_model_with_llm(tmp_path):
    user_id = 456
    mock_llm = AsyncMock()
    # Mock the LLM to return a valid JSON response
    mock_response = '```json\n{"preferences": {"likes_python": true}, "communication_style": "direct", "expertise_level": "advanced", "recurring_topics": ["coding"]}\n```'
    mock_llm.chat_completion.return_value = mock_response

    model = UserModel(persist_dir=str(tmp_path), llm=mock_llm)

    asyncio.run(model.update_model(user_id, "I like python and I want direct answers. I am advanced in coding."))

    # Verify the model was updated
    data = model.get_model(user_id)
    assert data["expertise_level"] == "advanced"
    assert data["communication_style"] == "direct"
    assert "likes_python" in data["preferences"]
    assert "coding" in data["recurring_topics"]

    # Check that LLM was called
    mock_llm.chat_completion.assert_called_once()
