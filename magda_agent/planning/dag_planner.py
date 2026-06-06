import logging
from typing import List, Dict, Any, Set, Tuple

class DAGPlanner:
    """
    Utility class for resolving task dependencies and computing topological sorts
    for execution plans.
    """

    @staticmethod
    def get_executable_steps(plan_steps: List[Dict[str, Any]], completed_step_ids: Set[str]) -> List[Dict[str, Any]]:
        """
        Returns a list of steps that have all their dependencies met and are not yet completed.

        Args:
            plan_steps (List[Dict[str, Any]]): The list of step dictionaries. Each should have 'id' and 'dependencies'.
            completed_step_ids (Set[str]): A set of IDs of steps that have already been completed.

        Returns:
            List[Dict[str, Any]]: A list of steps ready for execution.
        """
        executable_steps = []
        for step in plan_steps:
            step_id = step.get("id")
            if not step_id or step_id in completed_step_ids:
                continue

            dependencies = step.get("dependencies", [])
            # A step is executable if all its dependencies are in completed_step_ids
            if all(dep in completed_step_ids for dep in dependencies):
                executable_steps.append(step)

        return executable_steps

    @staticmethod
    def topological_sort(plan_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns a topologically sorted list of steps.
        If a cycle is detected, it logs an error and returns an empty list or raises an exception.
        For simplicity, this raises ValueError on cycles.

        Args:
            plan_steps (List[Dict[str, Any]]): The list of step dictionaries.

        Returns:
            List[Dict[str, Any]]: The sorted steps.
        """
        # Build adjacency list
        adj: Dict[str, List[str]] = {step.get("id", ""): [] for step in plan_steps if step.get("id")}
        in_degree: Dict[str, int] = {step.get("id", ""): 0 for step in plan_steps if step.get("id")}

        step_map = {step.get("id"): step for step in plan_steps if step.get("id")}

        for step in plan_steps:
            step_id = step.get("id")
            if not step_id:
                continue
            for dep in step.get("dependencies", []):
                if dep in adj:
                    adj[dep].append(step_id)
                    in_degree[step_id] += 1
                else:
                    logging.warning(f"Dependency {dep} for step {step_id} not found in plan steps.")

        # Kahn's algorithm
        queue = [node for node, deg in in_degree.items() if deg == 0]
        sorted_ids = []

        while queue:
            node = queue.pop(0)
            sorted_ids.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_ids) != len(adj):
            raise ValueError("Cycle detected in plan dependencies")

        return [step_map[node] for node in sorted_ids]
