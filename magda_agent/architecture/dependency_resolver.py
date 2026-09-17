import logging
from typing import List, Dict, Any, Set, Optional

logger = logging.getLogger(__name__)


class TaskDependencyResolver:
    """
    Hierarchical Planner Subagent Dependency Resolver.
    Extracts logic for resolving task dependencies, sorting DAG steps, and grouping independent
    parallel tasks into execution layers to be passed to Magentic-One / Agent Teams subagents.
    """

    @staticmethod
    def _extract_step_id(step: Dict[str, Any]) -> str:
        step_id = step.get("id") or step.get("task_id")
        if not step_id or not isinstance(step_id, str):
            raise ValueError("Each step must specify a non-empty string 'id' or 'task_id'.")
        return step_id

    @classmethod
    def resolve_execution_layers(
        cls, plan_steps: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Computes the parallel execution layers of a DAG plan using Kahn's algorithm.
        Each layer contains independent tasks whose dependencies are met by previous layers.

        Args:
            plan_steps (List[Dict[str, Any]]): List of step dictionaries with 'id'/'task_id' and 'dependencies'.

        Returns:
            List[List[Dict[str, Any]]]: Grouped execution layers of independent steps.

        Raises:
            ValueError: If a cycle, missing dependency, or invalid step structure is detected.
        """
        if not plan_steps:
            return []

        adj: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}
        step_map: Dict[str, Dict[str, Any]] = {}

        for step in plan_steps:
            step_id = cls._extract_step_id(step)
            if step_id in step_map:
                raise ValueError(f"Duplicate step ID '{step_id}' found in plan.")
            adj[step_id] = []
            in_degree[step_id] = 0
            step_map[step_id] = step

        for step in plan_steps:
            step_id = cls._extract_step_id(step)
            deps = step.get("dependencies")
            if deps is None:
                deps = []
            if not isinstance(deps, (list, tuple, set)):
                raise ValueError(f"Dependencies for step '{step_id}' must be a list or iterable.")

            for dep in deps:
                if not isinstance(dep, str):
                    raise ValueError(f"Dependency ID '{dep}' for step '{step_id}' must be a string.")
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

    @classmethod
    def topological_sort(cls, plan_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns a topologically sorted list of steps.

        Args:
            plan_steps (List[Dict[str, Any]]): The plan steps.

        Returns:
            List[Dict[str, Any]]: Flattened topologically sorted steps.
        """
        layers = cls.resolve_execution_layers(plan_steps)
        sorted_steps: List[Dict[str, Any]] = []
        for layer in layers:
            sorted_steps.extend(layer)
        return sorted_steps

    @classmethod
    def get_executable_steps(
        cls, plan_steps: List[Dict[str, Any]], completed_step_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Returns steps that have all dependencies met in `completed_step_ids` and are not yet completed.

        Args:
            plan_steps (List[Dict[str, Any]]): The plan steps.
            completed_step_ids (Set[str]): Set of step IDs that have finished execution.

        Returns:
            List[Dict[str, Any]]: Steps ready for execution.
        """
        executable_steps: List[Dict[str, Any]] = []
        for step in plan_steps:
            try:
                step_id = cls._extract_step_id(step)
            except ValueError:
                continue

            if step_id in completed_step_ids:
                continue

            dependencies = step.get("dependencies") or []
            all_met = all(dep in completed_step_ids for dep in dependencies)
            if all_met:
                executable_steps.append(step)

        return executable_steps


# Alias for compatibility
DependencyResolver = TaskDependencyResolver
