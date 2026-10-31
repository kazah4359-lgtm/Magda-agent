import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from magda_agent.scheduler.memory_compaction_cron_v2 import (
    MemoryCompactionCronV2,
    OpenClawMemoryCompactionCronV2,
    schedule_openclaw_memory_compaction_v2,
)
from magda_agent.scheduler.cron_scheduler_v2 import CronSchedulerV2
from magda_agent.memory.episodic import EpisodicMemory


@pytest.fixture
def mock_scheduler():
    scheduler = MagicMock(spec=CronSchedulerV2)
    return scheduler


@pytest.fixture
def memory():
    # Use EphemeralClient for isolated testing and clean up collection before/after
    mem = EpisodicMemory(persist_directory=":memory:")
    # Reset collection for clean per-test state
    try:
        mem.client.delete_collection("episodic_memory")
    except Exception:
        pass
    mem.collection = mem.client.get_or_create_collection("episodic_memory")
    yield mem
    try:
        mem.client.delete_collection("episodic_memory")
    except Exception:
        pass


def test_schedule_job_cron_scheduler_v2(mock_scheduler, memory):
    """Test scheduling memory compaction job with CronSchedulerV2 (add_task)."""
    compactor = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=mock_scheduler,
        cron_expr="0 3 * * *",
    )
    compactor.schedule_job()

    mock_scheduler.add_task.assert_called_once_with(
        "openclaw_memory_compaction_v2",
        "0 3 * * *",
        compactor.run_compaction,
    )


def test_schedule_job_duck_typing(memory):
    """Test duck typing for alternative schedulers (register_task, schedule)."""
    # Scheduler with register_task
    task_manager = MagicMock()
    del task_manager.add_task
    compactor_tm = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=task_manager,
        cron_expr="0 4 * * *",
        task_name="custom_task",
    )
    compactor_tm.schedule_job()
    task_manager.register_task.assert_called_once_with("custom_task", "0 4 * * *", compactor_tm.run_compaction)

    # Scheduler with schedule
    cron_sched = MagicMock()
    del cron_sched.add_task
    del cron_sched.register_task
    compactor_cs = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=cron_sched,
        cron_expr="0 2 * * *",
    )
    compactor_cs.schedule_job()
    cron_sched.schedule.assert_called_once_with("0 2 * * *", compactor_cs.run_compaction, name="openclaw_memory_compaction_v2")

    # Unsupported scheduler raises AttributeError
    empty_sched = object()
    compactor_err = MemoryCompactionCronV2(episodic_memory=memory, scheduler=empty_sched)
    with pytest.raises(AttributeError, match="Provided scheduler does not support"):
        compactor_err.schedule_job()


def test_schedule_convenience_function(mock_scheduler, memory):
    """Test the convenience function schedule_openclaw_memory_compaction_v2."""
    job = schedule_openclaw_memory_compaction_v2(
        episodic_memory=memory,
        scheduler=mock_scheduler,
        cron_expr="0 3 * * *",
        threshold=5,
        batch_size=5,
    )
    assert isinstance(job, OpenClawMemoryCompactionCronV2)
    mock_scheduler.add_task.assert_called_once()


@pytest.mark.asyncio
async def test_compaction_skipped_below_threshold(mock_scheduler, memory):
    """Test that compaction is skipped when active events are below threshold."""
    # Store fewer events than threshold
    for i in range(3):
        memory.store_event(f"Event {i}")

    compactor = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=mock_scheduler,
        threshold=5,
        batch_size=5,
    )

    result = await compactor.run_compaction()

    assert result["status"] == "skipped"
    assert result["reason"] == "below_threshold"
    assert result["events_count"] == 3
    assert result["compacted_count"] == 0

    # Ensure no events were decayed
    events = memory.get_all_events(include_decayed=False)
    assert len(events) == 3


@pytest.mark.asyncio
async def test_compaction_with_llm_generate(mock_scheduler, memory):
    """Test compaction execution when LLM client provides generate()."""
    for i in range(6):
        memory.store_event(f"User discussed project topic {i}")

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value="Synthesized discussion on topics 0 to 4.")

    compactor = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=mock_scheduler,
        llm_client=mock_llm,
        threshold=5,
        batch_size=5,
    )

    result = await compactor.run_compaction()

    assert result["status"] == "success"
    assert result["compacted_count"] == 5
    assert result["summary"] == "Synthesized discussion on topics 0 to 4."
    mock_llm.generate.assert_called_once()

    # Active events should now be 1 remaining original + 1 summary event = 2
    active_events = memory.get_all_events(include_decayed=False)
    assert len(active_events) == 2

    # Verify summary metadata
    summary_events = [e for e in active_events if e["metadata"].get("type") == "summary"]
    assert len(summary_events) == 1
    assert summary_events[0]["text"] == "Synthesized discussion on topics 0 to 4."
    assert summary_events[0]["metadata"]["compacted_count"] == 5

    # Check that decayed events are present when include_decayed=True
    all_events = memory.get_all_events(include_decayed=True)
    decayed_events = [e for e in all_events if e["metadata"].get("decayed") is True]
    assert len(decayed_events) == 5


@pytest.mark.asyncio
async def test_compaction_with_chat_completion(mock_scheduler, memory):
    """Test compaction when LLM client supports chat_completion."""
    for i in range(4):
        memory.store_event(f"Task note {i}")

    mock_llm = MagicMock(spec=["chat_completion"])
    mock_llm.chat_completion = AsyncMock(return_value="Consolidated summary via chat.")

    compactor = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=mock_scheduler,
        llm_client=mock_llm,
        threshold=3,
        batch_size=3,
    )

    result = await compactor.run_compaction()

    assert result["status"] == "success"
    assert result["compacted_count"] == 3
    assert result["summary"] == "Consolidated summary via chat."
    mock_llm.chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_compaction_fallback_heuristic_when_no_llm(mock_scheduler, memory):
    """Test fallback heuristic summarization when no LLM is provided."""
    for i in range(5):
        memory.store_event(f"Log event item {i}")

    compactor = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=mock_scheduler,
        llm_client=None,
        threshold=4,
        batch_size=4,
    )

    result = await compactor.run_compaction()

    assert result["status"] == "success"
    assert result["compacted_count"] == 4
    assert "Compacted summary of 4 events" in result["summary"]

    active_events = memory.get_all_events(include_decayed=False)
    assert len(active_events) == 2  # 1 uncompacted + 1 summary


@pytest.mark.asyncio
async def test_compaction_llm_exception_resilience(mock_scheduler, memory):
    """Test that LLM exceptions fall back gracefully to heuristic summary."""
    for i in range(5):
        memory.store_event(f"Error test event {i}")

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM API down"))

    compactor = MemoryCompactionCronV2(
        episodic_memory=memory,
        scheduler=mock_scheduler,
        llm_client=mock_llm,
        threshold=4,
        batch_size=4,
    )

    result = await compactor.run_compaction()

    assert result["status"] == "success"
    assert result["compacted_count"] == 4
    assert "Compacted summary of 4 events" in result["summary"]


@pytest.mark.asyncio
async def test_compaction_error_handling(mock_scheduler):
    """Test that unexpected errors in episodic_memory are caught and returned."""
    mock_memory = MagicMock()
    mock_memory.get_all_events.side_effect = Exception("Database failure")

    compactor = MemoryCompactionCronV2(
        episodic_memory=mock_memory,
        scheduler=mock_scheduler,
    )

    result = await compactor.run_compaction()
    assert result["status"] == "error"
    assert "Database failure" in result["error"]
    assert result["compacted_count"] == 0
