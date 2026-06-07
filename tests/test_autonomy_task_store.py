import os

import pytest

from magda_agent.autonomy.task_store import TaskStore, TaskStatus


@pytest.fixture
def store_path(tmp_path):
    return os.path.join(str(tmp_path), "tasks.json")


@pytest.mark.asyncio
async def test_add_and_get(store_path):
    store = TaskStore(path=store_path)
    task = await store.add_task("do a thing", user_id=42, max_iterations=5)
    assert task.status == TaskStatus.QUEUED
    fetched = await store.get(task.id)
    assert fetched is not None
    assert fetched.goal == "do a thing"
    assert fetched.user_id == 42
    assert fetched.max_iterations == 5


@pytest.mark.asyncio
async def test_claim_next_queued_is_fifo_and_marks_running(store_path):
    store = TaskStore(path=store_path)
    first = await store.add_task("first")
    await store.add_task("second")
    claimed = await store.claim_next_queued()
    assert claimed.id == first.id
    assert claimed.status == TaskStatus.RUNNING
    # second claim returns the other queued task
    claimed2 = await store.claim_next_queued()
    assert claimed2.goal == "second"
    # nothing left to claim
    assert await store.claim_next_queued() is None


@pytest.mark.asyncio
async def test_list_filters_by_user(store_path):
    store = TaskStore(path=store_path)
    await store.add_task("a", user_id=1)
    await store.add_task("b", user_id=2)
    assert len(await store.list()) == 2
    assert len(await store.list(user_id=1)) == 1


@pytest.mark.asyncio
async def test_cancel_queued_goes_straight_to_cancelled(store_path):
    store = TaskStore(path=store_path)
    task = await store.add_task("x")
    assert await store.request_cancel(task.id) is True
    assert (await store.get(task.id)).status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancel_running_sets_cancelling(store_path):
    store = TaskStore(path=store_path)
    task = await store.add_task("x")
    await store.claim_next_queued()
    assert await store.request_cancel(task.id) is True
    assert (await store.get(task.id)).status == TaskStatus.CANCELLING


@pytest.mark.asyncio
async def test_pause_and_resume(store_path):
    store = TaskStore(path=store_path)
    task = await store.add_task("x")
    await store.claim_next_queued()
    assert await store.request_pause(task.id) is True
    assert (await store.get(task.id)).status == TaskStatus.PAUSING
    await store.update(task.id, status=TaskStatus.PAUSED)
    assert await store.resume(task.id) is True
    assert (await store.get(task.id)).status == TaskStatus.QUEUED


@pytest.mark.asyncio
async def test_persistence_and_requeue_of_running(store_path):
    store = TaskStore(path=store_path)
    task = await store.add_task("persist me")
    await store.claim_next_queued()  # now RUNNING
    await store.append_progress(task.id, "did something")

    # New store instance reading the same file: running tasks become queued again.
    store2 = TaskStore(path=store_path)
    reloaded = await store2.get(task.id)
    assert reloaded is not None
    assert reloaded.status == TaskStatus.QUEUED
    assert len(reloaded.progress) == 1
