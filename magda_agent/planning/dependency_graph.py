import logging
from typing import List, Dict, Any, Set, Optional

class DependencyGraph:
    """
    Utility class for resolving task dependencies and computing topological sorts
    for execution plans. Implements the dependency graph task system for Claude multi-agent orchestration.
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
        executable_steps: List[Dict[str, Any]] = []
        for step in plan_steps:
            step_id: Optional[str] = step.get("id")
            if not step_id or step_id in completed_step_ids:
                continue

            dependencies: List[str] = step.get("dependencies", [])
            # A step is executable if all its dependencies are in completed_step_ids
            all_met: bool = all(dep in completed_step_ids for dep in dependencies)
            if all_met:
                executable_steps.append(step)

        return executable_steps

    @staticmethod
    def topological_sort(plan_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Returns a topologically sorted list of steps using Kahn's algorithm.
        Raises ValueError if a cycle is detected.

        Args:
            plan_steps (List[Dict[str, Any]]): The list of step dictionaries.

        Returns:
            List[Dict[str, Any]]: The sorted steps.
        """
        # Build adjacency list
        adj: Dict[str, List[str]] = {step.get("id", ""): [] for step in plan_steps if step.get("id")}
        in_degree: Dict[str, int] = {step.get("id", ""): 0 for step in plan_steps if step.get("id")}

        step_map: Dict[str, Dict[str, Any]] = {step.get("id", ""): step for step in plan_steps if step.get("id")}

        for step in plan_steps:
            step_id: Optional[str] = step.get("id")
            if not step_id:
                continue
            dependencies: List[str] = step.get("dependencies", [])
            for dep in dependencies:
                if dep in adj:
                    adj[dep].append(step_id)
                    in_degree[step_id] += 1
                else:
                    logging.warning(f"Dependency {dep} for step {step_id} not found in plan steps.")

        # Kahn's algorithm
        queue: List[str] = [node for node, deg in in_degree.items() if deg == 0]
        sorted_ids: List[str] = []

        while queue:
            node: str = queue.pop(0)
            sorted_ids.append(node)
            neighbors: List[str] = adj.get(node, [])
            for neighbor in neighbors:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_ids) != len(adj):
            raise ValueError("Cycle detected in plan dependencies")

        return [step_map[node] for node in sorted_ids]

    @staticmethod
    def get_execution_layers(plan_steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Computes the execution layers of the task graph. Each layer contains tasks that can be
        executed in parallel (i.e. all their dependencies are in previous layers).
        Raises ValueError if a cycle is detected.

        Args:
            plan_steps (List[Dict[str, Any]]): The list of step dictionaries.

        Returns:
            List[List[Dict[str, Any]]]: A list of layers, where each layer is a list of steps.
        """
        adj: Dict[str, List[str]] = {step.get("id", ""): [] for step in plan_steps if step.get("id")}
        in_degree: Dict[str, int] = {step.get("id", ""): 0 for step in plan_steps if step.get("id")}
        step_map: Dict[str, Dict[str, Any]] = {step.get("id", ""): step for step in plan_steps if step.get("id")}

        for step in plan_steps:
            step_id: Optional[str] = step.get("id")
            if not step_id:
                continue
            dependencies: List[str] = step.get("dependencies", [])
            for dep in dependencies:
                if dep in adj:
                    adj[dep].append(step_id)
                    in_degree[step_id] += 1
                else:
                    logging.warning(f"Dependency {dep} for step {step_id} not found in plan steps.")

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

    @staticmethod
    def get_critical_path(plan_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Finds the critical path (longest dependency chain) in the task graph.
        Assumes all steps take unit time.

        Args:
            plan_steps (List[Dict[str, Any]]): The list of step dictionaries.

        Returns:
            List[Dict[str, Any]]: The sequence of steps that form the longest path.
        """
        # First ensure there are no cycles and get a valid topological sort
        sorted_steps = DependencyGraph.topological_sort(plan_steps)
        step_map = {step.get("id", ""): step for step in plan_steps if step.get("id")}

        # Create adjacency list for dependencies (forward edges)
        adj: Dict[str, List[str]] = {step.get("id", ""): [] for step in plan_steps if step.get("id")}
        for step in plan_steps:
            step_id: Optional[str] = step.get("id")
            if not step_id:
                continue
            for dep in step.get("dependencies", []):
                if dep in adj:
                    adj[dep].append(step_id)

        # Distances and predecessors
        dist: Dict[str, int] = {step.get("id", ""): 0 for step in plan_steps if step.get("id")}
        prev: Dict[str, Optional[str]] = {step.get("id", ""): None for step in plan_steps if step.get("id")}

        # Process in topological order
        for step in sorted_steps:
            u = step.get("id", "")
            for v in adj.get(u, []):
                if dist[u] + 1 > dist[v]:
                    dist[v] = dist[u] + 1
                    prev[v] = u

        # Find the node with the maximum distance
        max_dist = -1
        end_node = None
        for node, d in dist.items():
            if d > max_dist:
                max_dist = d
                end_node = node

        if not end_node:
            return []

        # Reconstruct path
        path = []
        curr = end_node
        while curr is not None:
            path.append(step_map[curr])
            curr = prev[curr]

        path.reverse()
        return path
