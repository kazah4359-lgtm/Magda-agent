import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.letta_routine_v1 import LettaVirtualContextRoutineBuilder, LettaRoutineBuilderV1


@pytest.mark.asyncio
async def test_extract_patterns_heuristically():
    builder = LettaVirtualContextRoutineBuilder()
    records = [
        "Routine: Daily Standup\n- Check email\n- Review board\n- Post update",
        "Open terminal then run tests -> inspect coverage",
    ]
    patterns = builder.extract_patterns_heuristically(records)
    assert len(patterns) >= 2
    assert any(p["name"] == "Daily Standup" for p in patterns)
    assert any(p["name"] == "Sequential Action Routine" for p in patterns)


@pytest.mark.asyncio
async def test_extract_and_store_routines_with_mocked_memory():
    mock_episodic = MagicMock()
    mock_episodic.recall_events.return_value = [
        "Workflow: Code Review\n1. Fetch PR\n2. Run linter\n3. Leave comments"
    ]
    mock_procedural = MagicMock()

    builder = LettaVirtualContextRoutineBuilder(
        procedural_memory=mock_procedural,
        episodic_memory=mock_episodic,
        user_id=123,
    )

    stored = await builder.extract_and_store_routines()

    mock_episodic.recall_events.assert_called_once_with(
        query="routine procedure workflow step action", top_k=10, user_id=123
    )
    assert len(stored) >= 1
    assert stored[0]["name"] == "Code Review"
    mock_procedural.store_procedure.assert_called_once()
    call_args = mock_procedural.store_procedure.call_args[1]
    assert call_args["name"] == "Code Review"
    assert call_args["user_id"] == 123


@pytest.mark.asyncio
async def test_async_context_manager():
    mock_episodic = MagicMock()
    mock_episodic.recall_events.return_value = [
        "Procedure: Backup DB\nStep 1: dump sql\nStep 2: upload s3"
    ]
    mock_procedural = MagicMock()

    async with LettaVirtualContextRoutineBuilder(
        procedural_memory=mock_procedural,
        episodic_memory=mock_episodic,
        user_id=456,
    ) as builder:
        assert isinstance(builder, LettaRoutineBuilderV1)

    mock_procedural.store_procedure.assert_called_once()
    call_args = mock_procedural.store_procedure.call_args[1]
    assert call_args["name"] == "Backup DB"
    assert call_args["user_id"] == 456


@pytest.mark.asyncio
async def test_llm_extraction_flow():
    mock_episodic = MagicMock()
    mock_episodic.recall_events.return_value = ["Event A happened", "Event B happened"]
    mock_procedural = MagicMock()
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(return_value="Step 1: Do A\nStep 2: Do B")

    builder = LettaVirtualContextRoutineBuilder(
        procedural_memory=mock_procedural,
        episodic_memory=mock_episodic,
        llm_client=mock_llm,
        user_id=789,
    )

    stored = await builder.extract_and_store_routines()
    assert len(stored) == 1
    assert stored[0]["name"] == "LLM Extracted Routine"
    assert "Step 1: Do A" in stored[0]["procedure"]
