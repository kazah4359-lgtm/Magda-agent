"""
Hierarchical Planner V3 module.

Inspired by the Claude Agent SDK (docs/trends.md): implements a Hierarchical Planner
for sub-agent delegation. Decomposes complex user requests into a graph of sub-tasks
and orchestrates sub-agent dispatch without executing real external sub-agents when mocked.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from magda_agent.architecture.dependency_graph_v2 import (
    ClaudeSubagentDependencyGraphV2,
    SubagentTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class HierarchicalPlannerV3:
    """
    Hierarchical Planner V3 for sub-agent task decomposition and DAG delegation.

    Manages complex request breakdown into sub-agent task graphs, supports hierarchical
    sub-task expansion, Kahn-based parallel layer execution, and status propagation.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        spawner: Optional[Any] = None,
        graph: Optional[ClaudeSubagentDependencyGraphV2] = None,
    ) -> None:
        """
        Initializes HierarchicalPlannerV3.

        Args:
            llm_client: Optional LLM client instance for dynamic task decomposition.
            spawner: Optional sub-agent spawner instance.
            graph: Optional pre-constructed dependency graph.
        """
        self.llm_client = llm_client
        self.spawner = spawner
        self.graph = graph or ClaudeSubagentDependencyGraphV2()

    def decompose_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        custom_plan: Optional[List[Union[SubagentTask, Dict[str, Any]]]] = None,
    ) -> ClaudeSubagentDependencyGraphV2:
        """
        Decomposes a complex user request into a dependency graph of sub-tasks.

        Args:
            request: The high-level request string.
            context: Additional execution context.
            custom_plan: Optional list of task dicts or SubagentTask instances.

        Returns:
            ClaudeSubagentDependencyGraphV2 containing the decomposed task DAG.
        """
        self.graph = ClaudeSubagentDependencyGraphV2()

        if custom_plan:
            for item in custom_plan:
                self.graph.add_task(item)
            return self.graph

        if self.llm_client and hasattr(self.llm_client, "generate_plan"):
            plan_data = self.llm_client.generate_plan(request, context)
            if isinstance(plan_data, list):
                for item in plan_data:
                    self.graph.add_task(item)
                return self.graph

        # Default standard 3-stage heuristic decomposition (e.g., Analysis -> Execution -> Verification)
        analysis_task = SubagentTask(
            task_id="task_analysis",
            description=f"Analyze requirements for: {request}",
            required_capability="analyzer",
            input_data={"request": request, "context": context or {}},
        )
        execution_task = SubagentTask(
            task_id="task_execution",
            description=f"Execute core implementation for: {request}",
            dependencies={"task_analysis"},
            required_capability="executor",
            input_data={"request": request},
        )
        verification_task = SubagentTask(
            task_id="task_verification",
            description=f"Verify result for: {request}",
            dependencies={"task_execution"},
            required_capability="evaluator",
            input_data={"request": request},
        )

        self.graph.add_task(analysis_task)
        self.graph.add_task(execution_task)
        self.graph.add_task(verification_task)

        return self.graph

    def expand_subtask(
        self,
        parent_task_id: str,
        child_subtasks: List[Union[SubagentTask, Dict[str, Any]]],
    ) -> List[SubagentTask]:
        """
        Hierarchically decomposes a parent task into a set of finer-grained child sub-tasks.
        Replaces or links the parent task with child sub-tasks while maintaining dependency constraints.

        Args:
            parent_task_id: Task ID of the task being expanded.
            child_subtasks: List of child task specs or instances.

        Returns:
            List of newly added SubagentTask instances.
        """
        parent_task = self.graph.get_task(parent_task_id)
        if not parent_task:
            raise ValueError(f"Parent task '{parent_task_id}' not found in graph.")

        parent_deps = set(parent_task.dependencies)

        # Identify tasks that depended on the parent task
        dependent_tasks = [
            t for t in self.graph.get_all_tasks() if parent_task_id in t.dependencies
        ]

        # Remove parent task from graph
        self.graph.remove_task(parent_task_id)

        added_tasks: List[SubagentTask] = []
        child_ids: List[str] = []

        for item in child_subtasks:
            if isinstance(item, dict):
                task_obj = SubagentTask.from_dict(item)
            elif isinstance(item, SubagentTask):
                task_obj = item
            else:
                raise TypeError("child_subtasks items must be SubagentTask or dict.")

            # If child task has no explicit dependencies, inherit parent's dependencies
            if not task_obj.dependencies:
                task_obj.dependencies = set(parent_deps)

            added_task = self.graph.add_task(task_obj)
            added_tasks.append(added_task)
            child_ids.append(added_task.task_id)

        # Re-link downstream dependent tasks to depend on leaf child tasks
        leaf_child_ids = set(child_ids)
        for child in added_tasks:
            # A child is not a leaf if another child depends on it
            for other_child in added_tasks:
                if child.task_id in other_child.dependencies:
                    leaf_child_ids.discard(child.task_id)

        for dep_task in dependent_tasks:
            dep_task.dependencies.discard(parent_task_id)
            dep_task.dependencies.update(leaf_child_ids)

        return added_tasks

    def validate_graph(self) -> Tuple[bool, List[str]]:
        """
        Validates the structure of the DAG.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_error_messages)
        """
        return self.graph.validate_graph()

    def get_execution_layers(self) -> List[List[SubagentTask]]:
        """
        Computes parallel execution layers for independent sub-tasks.

        Returns:
            List[List[SubagentTask]]: Parallel execution layers.
        """
        return self.graph.get_execution_layers()

    def topological_sort(self) -> List[SubagentTask]:
        """
        Returns a topologically sorted sequence of sub-tasks.

        Returns:
            List[SubagentTask]: Ordered sub-tasks.
        """
        return self.graph.topological_sort()

    async def execute_plan_async(
        self,
        dispatch_fn: Optional[Callable[[SubagentTask, Dict[str, Any]], Any]] = None,
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates sub-agent execution layer by layer without executing real sub-agents.

        Args:
            dispatch_fn: Optional async or sync callback to execute/mock sub-agent dispatch.
                         Signature: dispatch_fn(task: SubagentTask, context: Dict[str, Any]) -> Any
            base_context: Base context passed to sub-agents.

        Returns:
            Summary dict with execution results, status, outputs, and execution order.
        """
        is_valid, errors = self.validate_graph()
        if not is_valid:
            return {
                "status": "failed",
                "error": f"Invalid task graph: {'; '.join(errors)}",
                "completed_tasks": [],
                "outputs": {},
            }

        context = dict(base_context or {})
        completed_task_ids: Set[str] = set()
        execution_order: List[str] = []
        task_outputs: Dict[str, Any] = {}
        failed_tasks: Set[str] = set()

        while True:
            executable_tasks = self.graph.get_executable_tasks(completed_task_ids)
            if not executable_tasks:
                break

            # Execute all currently ready independent tasks concurrently
            async def run_single_task(task: SubagentTask) -> Tuple[SubagentTask, Any, Optional[Exception]]:
                task.status = TaskStatus.RUNNING
                try:
                    if dispatch_fn:
                        res = dispatch_fn(task, context)
                        if hasattr(res, "__await__"):
                            res = await res
                    elif self.spawner and hasattr(self.spawner, "spawn"):
                        res = self.spawner.spawn(task=task.description, context=context)
                        if hasattr(res, "__await__"):
                            res = await res
                    else:
                        res = f"Simulated result for task '{task.task_id}': {task.description}"
                    return task, res, None
                except Exception as exc:
                    return task, None, exc

            results = await asyncio.gather(*(run_single_task(t) for t in executable_tasks))

            for task, res, exc in results:
                execution_order.append(task.task_id)
                if exc:
                    task_outputs[task.task_id] = str(exc)
                    failed_tasks.add(task.task_id)
                    self.graph.mark_task_failed(task.task_id, str(exc))
                else:
                    task_outputs[task.task_id] = res
                    completed_task_ids.add(task.task_id)
                    self.graph.mark_task_completed(task.task_id, res)
                    context[f"output_{task.task_id}"] = res

        all_tasks = self.graph.get_all_tasks()
        all_completed = all(t.status == TaskStatus.COMPLETED for t in all_tasks)

        return {
            "status": "completed" if all_completed else "failed",
            "completed_tasks": list(completed_task_ids),
            "failed_tasks": list(failed_tasks),
            "execution_order": execution_order,
            "outputs": task_outputs,
            "summary": self.get_plan_summary(),
        }

    def get_plan_summary(self) -> Dict[str, Any]:
        """
        Returns a summary dictionary of current graph state and tasks.
        """
        all_tasks = self.graph.get_all_tasks()
        return {
            "total_tasks": len(all_tasks),
            "completed": sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in all_tasks if t.status == TaskStatus.FAILED),
            "blocked": sum(1 for t in all_tasks if t.status == TaskStatus.BLOCKED),
            "pending": sum(1 for t in all_tasks if t.status == TaskStatus.PENDING),
            "tasks": [t.to_dict() for t in all_tasks],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes planner state to a dictionary representation."""
        return {
            "summary": self.get_plan_summary(),
            "layers": [
                [t.to_dict() for t in layer]
                for layer in self.get_execution_layers()
            ],
        }
