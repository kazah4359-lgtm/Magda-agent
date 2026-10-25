import pytest
from unittest.mock import AsyncMock, MagicMock
from magda_agent.memory.letta_routine_v2 import (
    LettaVirtualContextRoutineBuilderV2,
    LettaRoutineBuilderV2,
)


@pytest.mark.asyncio
async def test_extract_patterns_heuristically_v2():
    builder = LettaVirtualContextRoutineBuilderV2()
    stream = [
        "Routine: Deployment Pipeline\n- Pull latest main\n- Run tests\n- Deploy to prod",
        {"role": "user", "content": "Execute build => verify image => push registry"},
    ]
    patterns = builder.extract_patterns_heuristically(stream)
    assert len(patterns) >= 2
    assert any(p["name"] == "Deployment Pipeline" for p in patterns)
    assert any(p["name"] == "Sequential Workflow Routine V2" for p in patterns)


@pytest.mark.asyncio
async def test_process_context_stream_stores_procedural_memory():
    mock_procedural = MagicMock()
    builder = LettaVirtualContextRoutineBuilderV2(
        procedural_memory=mock_procedural,
        user_id=101,
    )

    stream = [
        "Workflow: Incident Response\nStep 1: Triage alert\nStep 2: Mitigate impact\nStep 3: Postmortem"
    ]
    stored = await builder.process_context_stream(stream, user_id=101)

    assert len(stored) == 1
    assert stored[0]["name"] == "Incident Response"
    mock_procedural.store_procedure.assert_called_once()
    kwargs = mock_procedural.store_procedure.call_args[1]
    assert kwargs["name"] == "Incident Response"
    assert kwargs["user_id"] == 101
    assert kwargs["metadata"]["version"] == "v2"


@pytest.mark.asyncio
async def test_async_context_manager_v2():
    mock_episodic = MagicMock()
    mock_episodic.recall_events.return_value = [
        "Routine: Data Backup\n- Dump DB\n- Sync S3"
    ]
    mock_procedural = MagicMock()

    async with LettaVirtualContextRoutineBuilderV2(
        procedural_memory=mock_procedural,
        episodic_memory=mock_episodic,
        user_id=202,
    ) as builder:
        assert isinstance(builder, LettaRoutineBuilderV2)

    mock_procedural.store_procedure.assert_called_once()
    kwargs = mock_procedural.store_procedure.call_args[1]
    assert kwargs["name"] == "Data Backup"
    assert kwargs["user_id"] == 202


@pytest.mark.asyncio
async def test_llm_extraction_flow_v2():
    mock_episodic = MagicMock()
    mock_episodic.recall_events.return_value = ["User requested automated report generation"]
    mock_procedural = MagicMock()
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value="Step 1: Fetch data\nStep 2: Format PDF\nStep 3: Send email"
    )

    builder = LettaVirtualContextRoutineBuilderV2(
        procedural_memory=mock_procedural,
        episodic_memory=mock_episodic,
        llm_client=mock_llm,
        user_id=303,
    )

    stored = await builder.extract_and_store_routines()
    assert len(stored) == 1
    assert stored[0]["name"] == "LLM Extracted Routine V2"
    assert "Step 1: Fetch data" in stored[0]["procedure"]
    mock_procedural.store_procedure.assert_called_once()
