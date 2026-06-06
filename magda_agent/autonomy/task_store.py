"""Persistent, async-safe store for long-running autonomous tasks.

Stdlib-only (json + asyncio) so it can run anywhere the agent runs without extra
dependencies. State is persisted to a single JSON file and written atomically so
in-flight tasks survive a process restart.
"""
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class TaskStatus:
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"

    ACTIVE = {QUEUED, RUNNING, PAUSING, CANCELLING}
    TERMINAL = {CANCELLED, COMPLETED, FAILED}


@dataclass
class TaskRecord:
    goal: str
    user_id: Optional[int] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = TaskStatus.QUEUED
    max_iterations: int = 20
    iterations: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[str] = None
    error: Optional[str] = None
    progress: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> Dict[str, Any]:
        """A lightweight view without the (potentially large) progress log."""
        data = asdict(self)
        data["progress_count"] = len(self.progress)
        data.pop("progress", None)
        return data


class TaskStore:
    """JSON-file backed task store guarded by an asyncio lock."""

    def __init__(self, path: str = "./autonomy_tasks.json"):
        self.path = path
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            self._tasks = self._read_from_disk()
            # Tasks that were mid-flight when the process died are requeued.
            for task in self._tasks.values():
                if task.status in (TaskStatus.RUNNING, TaskStatus.PAUSING, TaskStatus.CANCELLING):
                    task.status = TaskStatus.QUEUED
            self._loaded = True

    def _read_from_disk(self) -> Dict[str, TaskRecord]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return {tid: TaskRecord(**rec) for tid, rec in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logging.error("Failed to load task store %s: %s", self.path, exc)
            return {}

    def _flush_locked(self) -> None:
        """Persist current state atomically. Caller must hold the lock."""
        tmp_path = f"{self.path}.tmp"
        data = {tid: rec.to_dict() for tid, rec in self._tasks.items()}
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    async def add_task(self, goal: str, user_id: Optional[int] = None, max_iterations: int = 20) -> TaskRecord:
        await self._ensure_loaded()
        async with self._lock:
            task = TaskRecord(goal=goal, user_id=user_id, max_iterations=max_iterations)
            self._tasks[task.id] = task
            self._flush_locked()
            return task

    async def get(self, task_id: str) -> Optional[TaskRecord]:
        await self._ensure_loaded()
        async with self._lock:
            return self._tasks.get(task_id)

    async def list(self, user_id: Optional[int] = None) -> List[TaskRecord]:
        await self._ensure_loaded()
        async with self._lock:
            tasks = list(self._tasks.values())
        if user_id is not None:
            tasks = [t for t in tasks if t.user_id == user_id]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    async def claim_next_queued(self) -> Optional[TaskRecord]:
        """Atomically pick the oldest queued task and mark it running."""
        await self._ensure_loaded()
        async with self._lock:
            queued = [t for t in self._tasks.values() if t.status == TaskStatus.QUEUED]
            if not queued:
                return None
            task = min(queued, key=lambda t: t.created_at)
            task.status = TaskStatus.RUNNING
            task.updated_at = time.time()
            self._flush_locked()
            return task

    async def update(self, task_id: str, **fields: Any) -> Optional[TaskRecord]:
        await self._ensure_loaded()
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for key, value in fields.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = time.time()
            self._flush_locked()
            return task

    async def append_progress(self, task_id: str, message: str, event: str = "info") -> None:
        await self._ensure_loaded()
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.progress.append({"ts": time.time(), "event": event, "message": message})
            task.updated_at = time.time()
            self._flush_locked()

    async def request_cancel(self, task_id: str) -> bool:
        await self._ensure_loaded()
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in TaskStatus.TERMINAL:
                return False
            task.status = TaskStatus.CANCELLING if task.status != TaskStatus.QUEUED else TaskStatus.CANCELLED
            task.updated_at = time.time()
            self._flush_locked()
            return True

    async def request_pause(self, task_id: str) -> bool:
        await self._ensure_loaded()
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status not in (TaskStatus.RUNNING, TaskStatus.QUEUED):
                return False
            task.status = TaskStatus.PAUSING if task.status == TaskStatus.RUNNING else TaskStatus.PAUSED
            task.updated_at = time.time()
            self._flush_locked()
            return True

    async def resume(self, task_id: str) -> bool:
        await self._ensure_loaded()
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status not in (TaskStatus.PAUSED, TaskStatus.PAUSING):
                return False
            task.status = TaskStatus.QUEUED
            task.updated_at = time.time()
            self._flush_locked()
            return True
