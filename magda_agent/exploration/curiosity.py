from typing import List, Dict, Any

class CuriosityExplorer:
    """
    Curiosity Explorer module.
    Responsible for proactively suggesting read-only exploration actions
    when the agent's boredom level reaches a certain threshold.
    """

    def __init__(self, boredom_threshold: float = 0.8) -> None:
        """
        Initializes the CuriosityExplorer.

        Args:
            boredom_threshold (float): The minimum boredom level (0.0 to 1.0)
                required to trigger an exploration suggestion.
        """
        self.boredom_threshold = boredom_threshold

    def should_explore(self, boredom: float) -> bool:
        """
        Determines whether exploration should be triggered based on current boredom.

        Args:
            boredom (float): The current boredom level of the agent.

        Returns:
            bool: True if boredom is at or above the threshold, False otherwise.
        """
        return boredom >= self.boredom_threshold

    def explore(self, workspace_context: Dict[str, Any] | None = None) -> List[str]:
        """
        Generates a list of safe, read-only exploration actions.
        These are proposed to the workspace when no other user tasks are pending.

        Args:
            workspace_context (Dict[str, Any] | None): Optional context from the global
                workspace to guide exploration. Currently unused but reserved for future.

        Returns:
            List[str]: A list of actionable string descriptions of tasks to explore.
        """
        if workspace_context is None:
            workspace_context = {}

        return [
            "Propose code review on recent PRs",
            "Read newly updated documentation",
            "Analyze current system architecture for missing components",
            "Check recent failing tests for patterns"
        ]
