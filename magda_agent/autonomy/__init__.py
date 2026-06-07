"""Long-running autonomous task execution.

This package adds a persistent task queue and a background worker that can keep
working a single goal over many planning/execution iterations (well beyond the
short per-message turn handled by `Consciousness.process_input`).
"""
from magda_agent.autonomy.task_store import TaskRecord, TaskStore, TaskStatus
from magda_agent.autonomy.executor import AutonomousExecutor

__all__ = ["TaskRecord", "TaskStore", "TaskStatus", "AutonomousExecutor"]
