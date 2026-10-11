from enum import Enum
from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any, Set, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Execution status of a subagent task node in the dependency graph."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class SubagentTask:
    """
    Represents a task node in the Claude subagent dependency graph.
    """
    task_id: str
    description: str = ""
    dependencies: Set[str] = field(default_factory=set)
    assigned_subagent: Optional[str] = None
    required_capability: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("Task ID must be a non-empty string.")
        if isinstance(self.dependencies, (list, tuple)):
            self.dependencies = set(self.dependencies)
        if isinstance(self.status, str) and not isinstance(self.status, TaskStatus):
            self.status = TaskStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the subagent task into a dictionary format."""
        return {
            "id": self.task_id,
            "task_id": self.task_id,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "assigned_subagent": self.assigned_subagent,
            "required_capability": self.required_capability,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubagentTask":
        """Factory method to construct a SubagentTask from a dictionary representation."""
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise ValueError("Dictionary must specify 'task_id' or 'id'.")

        raw_deps = data.get("dependencies", [])
        if isinstance(raw_deps, (list, tuple)):
            deps = set(raw_deps)
        elif isinstance(raw_deps, set):
            deps = raw_deps
        else:
            deps = set()

        raw_status = data.get("status", TaskStatus.PENDING)
        if isinstance(raw_status, str):
            status = TaskStatus(raw_status)
        else:
            status = raw_status

        return cls(
            task_id=str(task_id),
            description=str(data.get("description", "") or data.get("task", "")),
            dependencies=deps,
            assigned_subagent=data.get("assigned_subagent"),
            required_capability=data.get("required_capability"),
            status=status,
            input_data=dict(data.get("input_data", {})),
            output_data=data.get("output_data"),
            error=data.get("error"),
        )


class ClaudeSubagentDependencyGraphV2:
    """
    Claude Subagent Dependency Graph V2.
    Inspired by Claude Agent SDK Task system with dependency graphs.
    Provides DAG validation, execution order validation, parallel execution layer resolution,
    and task status propagation across subagents.
    """

    def __init__(self, tasks: Optional[List[Union[SubagentTask, Dict[str, Any]]]] = None) -> None:
        self._tasks: Dict[str, SubagentTask] = {}
        if tasks:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: Union[SubagentTask, Dict[str, Any]]) -> SubagentTask:
        """Adds a task node to the dependency graph."""
        if isinstance(task, dict):
            task_obj = SubagentTask.from_dict(task)
        elif isinstance(task, SubagentTask):
            task_obj = task
        else:
            raise TypeError("task must be a SubagentTask instance or dict.")

        if task_obj.task_id in self._tasks:
            raise ValueError(f"Duplicate task ID '{task_obj.task_id}' in dependency graph.")

        self._tasks[task_obj.task_id] = task_obj
        return task_obj

    def remove_task(self, task_id: str) -> None:
        """Removes a task from the graph by task_id."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            # Remove from dependencies of remaining tasks
            for task in self._tasks.values():
                task.dependencies.discard(task_id)

    def get_task(self, task_id: str) -> Optional[SubagentTask]:
        """Retrieves a task by task_id."""
        return self._tasks.get(task_id)

    def has_task(self, task_id: str) -> bool:
        """Checks if a task_id exists in the graph."""
        return task_id in self._tasks

    def get_all_tasks(self) -> List[SubagentTask]:
        """Returns all task nodes in the graph."""
        return list(self._tasks.values())

    def validate_graph(self) -> Tuple[bool, List[str]]:
        """
        Validates the structure of the DAG.
        Checks for missing dependencies, self-dependencies, and cycles.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_error_strings)
        """
        errors: List[str] = []

        # Check for missing dependencies and self-dependencies
        for task_id, task in self._tasks.items():
            for dep in task.dependencies:
                if dep == task_id:
                    errors.append(f"Task '{task_id}' cannot depend on itself.")
                elif dep not in self._tasks:
                    errors.append(f"Task '{task_id}' has non-existent dependency '{dep}'.")

        # Check for cycles using in-degree reduction (Kahn's algorithm)
        adj: Dict[str, List[str]] = {tid: [] for tid in self._tasks}
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}

        for task_id, task in self._tasks.items():
            for dep in task.dependencies:
                if dep in self._tasks and dep != task_id:
                    adj[dep].append(task_id)
                    in_degree[task_id] += 1

        queue: List[str] = [tid for tid, deg in in_degree.items() if deg == 0]
        processed_count = 0

        while queue:
            node = queue.pop(0)
            processed_count += 1
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if processed_count != len(self._tasks):
            errors.append("Cycle detected in subagent task dependencies.")

        return len(errors) == 0, errors

    def validate_execution_order(self, execution_order: List[str]) -> Tuple[bool, List[str]]:
        """
        Validates whether a given subagent task execution sequence satisfies all dependency constraints.

        Args:
            execution_order (List[str]): Order of task IDs proposed for execution.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_validation_errors)
        """
        errors: List[str] = []
        executed_so_far: Set[str] = set()

        for idx, task_id in enumerate(execution_order):
            if task_id not in self._tasks:
                errors.append(f"Execution order step {idx}: Task '{task_id}' does not exist in graph.")
                continue

            task = self._tasks[task_id]
            unmet_deps = task.dependencies - executed_so_far
            if unmet_deps:
                errors.append(
                    f"Task '{task_id}' executed at index {idx} before its dependencies: {sorted(list(unmet_deps))}."
                )

            executed_so_far.add(task_id)

        # Check for unexecuted tasks in graph
        missing_tasks = set(self._tasks.keys()) - set(execution_order)
        if missing_tasks:
            errors.append(f"Execution order is missing tasks from graph: {sorted(list(missing_tasks))}.")

        return len(errors) == 0, errors

    def get_executable_tasks(self, completed_task_ids: Optional[Set[str]] = None) -> List[SubagentTask]:
        """
        Returns all tasks ready for immediate subagent execution.

        A task is executable if its current status is PENDING or BLOCKED,
        and all its dependencies are satisfied.
        """
        if completed_task_ids is None:
            completed_task_ids = {
                tid for tid, task in self._tasks.items() if task.status == TaskStatus.COMPLETED
            }

        executable: List[SubagentTask] = []
        for task in self._tasks.values():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED):
                continue

            if task.dependencies.issubset(completed_task_ids):
                executable.append(task)

        return executable

    def get_execution_layers(self) -> List[List[SubagentTask]]:
        """
        Computes parallel execution layers using Kahn's algorithm.
        Each layer contains independent tasks ready to execute concurrently once prior layers finish.

        Raises:
            ValueError: If a cycle or invalid dependency structure is found.
        """
        is_valid, errors = self.validate_graph()
        if not is_valid:
            raise ValueError(f"Cannot compute execution layers for invalid graph: {'; '.join(errors)}")

        adj: Dict[str, List[str]] = {tid: [] for tid in self._tasks}
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}

        for task_id, task in self._tasks.items():
            for dep in task.dependencies:
                adj[dep].append(task_id)
                in_degree[task_id] += 1

        layers: List[List[SubagentTask]] = []
        queue: List[str] = [tid for tid, deg in in_degree.items() if deg == 0]

        while queue:
            current_layer_ids = list(queue)
            current_layer_tasks = [self._tasks[tid] for tid in current_layer_ids]
            layers.append(current_layer_tasks)
            queue = []

            for tid in current_layer_ids:
                for neighbor in adj.get(tid, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return layers

    def topological_sort(self) -> List[SubagentTask]:
        """Flattens execution layers into a topologically sorted task list."""
        layers = self.get_execution_layers()
        sorted_tasks: List[SubagentTask] = []
        for layer in layers:
            sorted_tasks.extend(layer)
        return sorted_tasks

    def mark_task_completed(self, task_id: str, output_data: Optional[Any] = None) -> List[SubagentTask]:
        """
        Marks a task as COMPLETED, records output data, and returns newly executable tasks.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Task '{task_id}' not found in dependency graph.")

        task = self._tasks[task_id]
        task.status = TaskStatus.COMPLETED
        if output_data is not None:
            task.output_data = output_data

        return self.get_executable_tasks()

    def mark_task_failed(self, task_id: str, error: Optional[str] = None) -> List[SubagentTask]:
        """
        Marks a task as FAILED, records error, and transitively marks downstream dependent tasks as BLOCKED.
        Returns the list of blocked subagent tasks.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Task '{task_id}' not found in dependency graph.")

        task = self._tasks[task_id]
        task.status = TaskStatus.FAILED
        if error is not None:
            task.error = str(error)

        # Transitive propagation of BLOCKED status to downstream tasks
        blocked_tasks: List[SubagentTask] = []

        def block_downstream(failed_id: str) -> None:
            for dep_task in self._tasks.values():
                if failed_id in dep_task.dependencies and dep_task.status not in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.SKIPPED,
                ):
                    dep_task.status = TaskStatus.BLOCKED
                    dep_task.error = f"Blocked due to failure of prerequisite task '{failed_id}'"
                    if dep_task not in blocked_tasks:
                        blocked_tasks.append(dep_task)
                    block_downstream(dep_task.task_id)

        block_downstream(task_id)
        return blocked_tasks

    def reset(self) -> None:
        """Resets status of all tasks in the graph to PENDING."""
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
            task.output_data = None
            task.error = None

    @classmethod
    def from_dict_list(cls, task_dicts: List[Dict[str, Any]]) -> "ClaudeSubagentDependencyGraphV2":
        """Factory constructor from a list of step/task dictionaries."""
        return cls(tasks=[SubagentTask.from_dict(d) for d in task_dicts])


# Alias for compatibility
DependencyGraphV2 = ClaudeSubagentDependencyGraphV2
