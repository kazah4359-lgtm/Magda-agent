import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from magda_agent.memory.letta_sync_v3 import (
    LettaAdvancedSyncV3,
    LettaVirtualContextAdvancedSyncV3,
)


@pytest.mark.asyncio
async def test_extract_patterns_heuristically_v3():
    sync_hook = LettaVirtualContextAdvancedSyncV3(storage_dir="/tmp/letta_test_extract")
    stream = [
        "Routine: Deployment Pipeline V3\n- Git pull origin main\n- Run pytest test suite\n- Deploy service container",
        {"role": "assistant", "content": "Fetch metrics => analyze threshold => trigger alert"},
    ]
    patterns = sync_hook.extract_patterns_heuristically(stream)
    assert len(patterns) >= 2
    assert any("Deployment Pipeline V3" in p["name"] for p in patterns)
    assert any("Sequential Workflow Routine V3" in p["name"] for p in patterns)


@pytest.mark.asyncio
async def test_process_context_stream_disk_sync_and_procedural_store(tmp_path):
    mock_procedural = MagicMock()
    storage_path = tmp_path / "sync_store"

    sync_hook = LettaVirtualContextAdvancedSyncV3(
        storage_dir=str(storage_path),
        procedural_memory=mock_procedural,
        user_id=777,
    )

    stream = [
        "Workflow: CI Pipeline Automation\nStep 1: Check out repo\nStep 2: Build wheel\nStep 3: Publish release"
    ]

    stored = await sync_hook.process_context_stream(stream, user_id=777, sync_to_disk=True)

    assert len(stored) == 1
    assert stored[0]["name"] == "CI Pipeline Automation"
    mock_procedural.store_procedure.assert_called_once()
    kwargs = mock_procedural.store_procedure.call_args[1]
    assert kwargs["name"] == "CI Pipeline Automation"
    assert kwargs["user_id"] == 777
    assert kwargs["metadata"]["version"] == "v3"

    synced_files = list(storage_path.glob("*.json"))
    assert len(synced_files) == 1
    with open(synced_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["name"] == "CI Pipeline Automation"
        assert data["user_id"] == 777
        assert data["version"] == "v3"

    loaded = sync_hook.load_synced_procedures(user_id=777)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "CI Pipeline Automation"


@pytest.mark.asyncio
async def test_mocked_disk_writes_and_context_updates():
    mock_procedural = MagicMock()
    sync_hook = LettaVirtualContextAdvancedSyncV3(
        storage_dir="/fake/disk/dir",
        procedural_memory=mock_procedural,
        user_id=888,
    )

    stream = [
        "Routine: Memory Recovery\n- Flush cache\n- Restart worker"
    ]

    with patch.object(
        LettaVirtualContextAdvancedSyncV3, "write_procedure_to_disk", new_callable=AsyncMock
    ) as mock_write:
        mock_write.return_value = Path("/fake/disk/dir/user_888_Memory_Recovery.json")

        stored = await sync_hook.process_context_stream(stream, user_id=888, sync_to_disk=True)

        assert len(stored) == 1
        mock_write.assert_awaited_once()
        args, kwargs = mock_write.call_args
        assert args[0]["name"] == "Memory Recovery"
        assert kwargs["user_id"] == 888

    # Verify context update hook
    sync_hook.on_context_update(["Routine: Hotfix Release\n- Patch code\n- Push tag"], user_id=888)
    assert any("Hotfix Release" in r["name"] for r in sync_hook.synced_procedures)


@pytest.mark.asyncio
async def test_async_context_manager_and_alias(tmp_path):
    mock_episodic = MagicMock()
    mock_episodic.recall_events.return_value = [
        "Routine: Database Compaction\n- Vacuum tables\n- Reindex B-trees"
    ]
    mock_procedural = MagicMock()
    storage_path = tmp_path / "sync_context_mgr"

    async with LettaVirtualContextAdvancedSyncV3(
        storage_dir=str(storage_path),
        procedural_memory=mock_procedural,
        episodic_memory=mock_episodic,
        user_id=999,
        auto_sync=True,
    ) as sync_instance:
        assert isinstance(sync_instance, LettaAdvancedSyncV3)

    mock_procedural.store_procedure.assert_called_once()
    kwargs = mock_procedural.store_procedure.call_args[1]
    assert kwargs["name"] == "Database Compaction"
    assert kwargs["user_id"] == 999

    persisted = list(storage_path.glob("*.json"))
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_llm_extraction_and_compact_hook(tmp_path):
    mock_episodic = MagicMock()
    mock_episodic.recall_events.return_value = ["User requested end-to-end telemetry sync"]
    mock_procedural = MagicMock()
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value="Step 1: Ingest metrics\nStep 2: Aggregate summaries\nStep 3: Ship telemetry"
    )

    storage_path = tmp_path / "llm_sync"
    sync_hook = LettaVirtualContextAdvancedSyncV3(
        storage_dir=str(storage_path),
        procedural_memory=mock_procedural,
        episodic_memory=mock_episodic,
        llm_client=mock_llm,
        user_id=123,
    )

    stored = await sync_hook.extract_and_sync_routines(user_id=123)
    assert len(stored) == 1
    assert stored[0]["name"] == "LLM Extracted Routine V3"
    assert "Step 1: Ingest metrics" in stored[0]["procedure"]
    mock_procedural.store_procedure.assert_called_once()

    # Test compact lifecycle hook
    class DummyItem:
        def __init__(self, content):
            self.content = content

    compact_items = [
        DummyItem("Workflow: Compacted Cleanout\n- Evict oldest cache\n- Persist state")
    ]
    returned_items = await sync_hook.compact(compact_items, metadata={"user_id": 123})
    assert len(returned_items) == 1
    assert len(list(storage_path.glob("*.json"))) >= 2
