import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, Set

from magda_agent.architecture.parallel_execution_v4 import ParallelSubagentManagerV4
from magda_agent.architecture.subagent_spawner_v4 import SubagentSpawnerV4

logger = logging.getLogger(__name__)


class MagenticOneDependencyRouterV4:
    """
    Implements a DAG-based orchestration router based on Microsoft's Magentic-One and Agent Teams trends.
    Topologically resolves task dependencies into parallel execution layers and routes independent
    sub-tasks concurrently to subagents operating in isolated git worktrees.
    """

    def __init__(
        self,
        spawner: Optional[SubagentSpawnerV4] = None,
        parallel_manager: Optional[ParallelSubagentManagerV4] = None,
    ) -> None:
        """
        Initializes the MagenticOneDependencyRouterV4.

        Args:
            spawner: Optional SubagentSpawnerV4 instance.
            parallel_manager: Optional ParallelSubagentManagerV4 instance.
        """
        self.spawner = spawner or SubagentSpawnerV4()
        self.parallel_manager = parallel_manager or ParallelSubagentManagerV4(spawner=self.spawner)

    def resolve_execution_layers(self, plan_steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Computes the execution layers of the task graph using Kahn's algorithm.
        Each layer contains independent tasks whose dependencies are met by prior layers.

        Args:
            plan_steps (List[Dict[str, Any]]): List of step dictionaries with 'id' and 'dependencies'.

        Returns:
            List[List[Dict[str, Any]]]: A list of execution layers.

        Raises:
            ValueError: If a cycle or invalid dependency structure is detected.
        """
        if not plan_steps:
            return []

        adj: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}
        step_map: Dict[str, Dict[str, Any]] = {}

        for step in plan_steps:
            step_id = step.get("id") or step.get("task_id")
            if not step_id:
                raise ValueError("Each step must specify an 'id' or 'task_id'.")
            adj[step_id] = []
            in_degree[step_id] = 0
            step_map[step_id] = step

        for step in plan_steps:
            step_id = step.get("id") or step.get("task_id")
            deps = step.get("dependencies")
            if deps is None:
                deps = []
            if not isinstance(deps, (list, tuple, set)):
                raise ValueError(f"Dependencies for step '{step_id}' must be a list or iterable.")

            for dep in deps:
                if dep in adj:
                    adj[dep].append(step_id)
                    in_degree[step_id] += 1
                else:
                    raise ValueError(f"Dependency '{dep}' for step '{step_id}' not found in plan steps.")

        layers: List[List[Dict[str, Any]]] = []
        queue: List[str] = [node for node, deg in in_degree.items() if deg == 0]
        processed_count = 0

        while queue:
            current_layer_nodes = list(queue)
            current_layer = [step_map[node] for node in current_layer_nodes]
            layers.append(current_layer)
            queue = []

            for node in current_layer_nodes:
                processed_count += 1
                for neighbor in adj.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        if processed_count != len(adj):
            raise ValueError("Cycle detected in plan dependencies")

        return layers

    def topological_sort(self, plan_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns a topologically sorted list of steps.

        Args:
            plan_steps (List[Dict[str, Any]]): The plan steps.

        Returns:
            List[Dict[str, Any]]: Flattened topologically sorted steps.
        """
        layers = self.resolve_execution_layers(plan_steps)
        sorted_steps: List[Dict[str, Any]] = []
        for layer in layers:
            sorted_steps.extend(layer)
        return sorted_steps

    async def route_dag_plan(
        self,
        plan_steps: List[Dict[str, Any]],
        base_context: Optional[List[Dict[str, Any]]] = None,
        agent_executor_factory: Optional[Callable[[], Any]] = None,
        merge_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Routes and executes a DAG plan. Resolves layers, dispatches independent sub-tasks in parallel
        via isolated worktrees, and propagates results.

        Args:
            plan_steps: The list of DAG step definitions.
            base_context: Shared context list.
            agent_executor_factory: Optional callable returning a subagent executor.
            merge_results: Whether to perform git branch merge for completed subagent tasks.

        Returns:
            Dict[str, Any] containing layer results, task outputs, merged state, and execution order.
        """
        if base_context is None:
            base_context = []

        layers = self.resolve_execution_layers(plan_steps)
        task_outputs: Dict[str, Any] = {}
        failed_tasks: Set[str] = set()
        execution_order: List[str] = []
        layer_results: List[List[Dict[str, Any]]] = []

        current_context = list(base_context)

        for layer_idx, layer in enumerate(layers):
            # Determine tasks in this layer that can run (dependencies not failed)
            executable_in_layer: List[Dict[str, Any]] = []
            skipped_in_layer: List[Dict[str, Any]] = []

            for step in layer:
                step_id = step.get("id") or step.get("task_id")
                deps = step.get("dependencies")
                if deps is None:
                    deps = []

                if any(dep in failed_tasks for dep in deps):
                    skipped_in_layer.append(step)
                    failed_tasks.add(step_id)
                    task_outputs[step_id] = RuntimeError(f"Skipped due to failed dependency in layer {layer_idx}")
                else:
                    executable_in_layer.append(step)

            if not executable_in_layer:
                layer_results.append([])
                continue

            tasks_descriptions = [
                step.get("description") or step.get("task") or str(step.get("id"))
                for step in executable_in_layer
            ]

            def default_executor_factory():
                class SimpleExecutor:
                    async def execute(self, ctx, **kwargs):
                        return {"status": "completed", "context_length": len(ctx)}
                return SimpleExecutor()

            factory = agent_executor_factory or default_executor_factory

            # Execute tasks in layer concurrently
            results = await self.parallel_manager.run_parallel_tasks(
                tasks=tasks_descriptions,
                base_context=current_context,
                agent_executor_factory=factory,
                merge_results=merge_results,
            )

            layer_outcomes: List[Dict[str, Any]] = []
            for step, result in zip(executable_in_layer, results):
                step_id = step.get("id") or step.get("task_id")
                execution_order.append(step_id)
                task_outputs[step_id] = result

                if isinstance(result, Exception):
                    failed_tasks.add(step_id)
                    logger.error(f"Task '{step_id}' failed: {result}")
                else:
                    # Append result summary to current context for subsequent layers
                    current_context.append({
                        "role": "system",
                        "content": f"Output from step '{step_id}': {result}"
                    })

                layer_outcomes.append({
                    "step_id": step_id,
                    "result": result if not isinstance(result, Exception) else str(result),
                    "status": "failed" if isinstance(result, Exception) else "success",
                })

            layer_results.append(layer_outcomes)

        merged_state = self.parallel_manager.merge_state(
            [res for res in task_outputs.values() if not isinstance(res, Exception)]
        )

        return {
            "layers_executed": len(layers),
            "layer_results": layer_results,
            "task_outputs": task_outputs,
            "merged_state": merged_state,
            "failed_tasks": list(failed_tasks),
            "execution_order": execution_order,
        }


# Alias for compatibility
DependencyRouterV4 = MagenticOneDependencyRouterV4
