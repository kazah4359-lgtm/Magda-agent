import fnmatch
from typing import List, Optional

class RiskSystem:
    """
    Amygdala/RiskSystem for change classification.
    Scores file changes and action requests to determine risk levels before auto-merge.
    """

    # Define risk rules according to cognitive architecture
    CRITICAL_PATTERNS = [
        "secrets/**",
        ".env*",
        "deployment/**",
    ]

    HIGH_PATTERNS = [
        ".github/workflows/**",
        "requirements.txt",
        "magda_agent/skills/registry.py", # skill registry changes
        "magda_agent/skills/system_execute_code.py", # sandbox changes
        "magda_agent/skills/omnichannel.py", # messaging providers
        "magda_agent/skills/internet_search.py", # network access
    ]

    LOW_PATTERNS = [
        "docs/**",
        "*.md",
    ]

    def __init__(self) -> None:
        """
        Initializes the RiskSystem.
        """
        pass

    def classify_file_change(self, filepath: str) -> str:
        """
        Classifies a single file path into a risk level.

        Args:
            filepath (str): The relative path of the file being changed.

        Returns:
            str: Risk level ("low", "medium", "high", "critical").
        """
        for pattern in self.CRITICAL_PATTERNS:
            if fnmatch.fnmatch(filepath, pattern) or filepath == pattern:
                return "critical"

        for pattern in self.HIGH_PATTERNS:
            if fnmatch.fnmatch(filepath, pattern) or filepath == pattern:
                return "high"

        for pattern in self.LOW_PATTERNS:
            if fnmatch.fnmatch(filepath, pattern) or filepath == pattern:
                return "low"

        # Default risk level for everything else
        return "medium"

    def classify_changes(self, filepaths: List[str]) -> str:
        """
        Classifies a list of file changes and returns the highest risk level.

        Args:
            filepaths (List[str]): List of file paths being changed.

        Returns:
            str: The overall risk level ("low", "medium", "high", "critical").
        """
        if not filepaths:
            return "low"

        risk_scores = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "critical": 3
        }

        max_risk = "low"
        for filepath in filepaths:
            risk = self.classify_file_change(filepath)
            if risk_scores[risk] > risk_scores[max_risk]:
                max_risk = risk

        return max_risk
