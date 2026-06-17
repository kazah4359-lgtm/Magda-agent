import pytest
from unittest.mock import MagicMock, AsyncMock
from magda_agent.planning.reflection_tasks import ReflectionTaskGenerator

@pytest.mark.asyncio
async def test_generate_tasks_from_reflection_success():
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='''
    [
      {
        "id": "fix-memory-leak-v1",
        "status": "todo",
        "area": "memory",
        "risk": "medium",
        "title": "Fix memory leak",
        "description": "The agent observed high memory usage. Need to fix it.",
        "allowed_paths": ["magda_agent/memory/storage.py", "tests/test_memory.py", "agent_tasks.json"],
        "acceptance": ["Memory usage remains stable."]
      }
    ]
    ''')

    generator = ReflectionTaskGenerator(llm_client=mock_llm)
    tasks = await generator.generate_tasks_from_reflection(
        "I noticed I am using too much memory.",
        ["Memory is expensive"],
        ["Storing everything forever"]
    )

    assert len(tasks) == 1
    assert tasks[0]["id"] == "fix-memory-leak-v1"
    assert tasks[0]["area"] == "memory"
    assert "agent_tasks.json" in tasks[0]["allowed_paths"]

@pytest.mark.asyncio
async def test_generate_tasks_from_reflection_invalid_json():
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='''
    [
      {
        "id": "missing-fields",
        "area": "memory"
      }
    ]
    ''')

    generator = ReflectionTaskGenerator(llm_client=mock_llm)
    tasks = await generator.generate_tasks_from_reflection(
        "Reflection", [], []
    )

    assert len(tasks) == 0

@pytest.mark.asyncio
async def test_generate_tasks_from_reflection_not_json():
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value='This is not a JSON list.')

    generator = ReflectionTaskGenerator(llm_client=mock_llm)
    tasks = await generator.generate_tasks_from_reflection(
        "Reflection", [], []
    )

    assert len(tasks) == 0
