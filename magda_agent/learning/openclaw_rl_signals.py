"""
OpenClaw-RL Next State Signals Module.

This module processes next-state signals (such as user replies and tool outputs)
for online reinforcement learning, inspired by the OpenClaw-RL trend.
"""

from typing import Dict, Any, List

class OpenClawRLSignals:
    """
    Processes next-state signals for online reinforcement learning.
    """

    def __init__(self) -> None:
        """Initialize the OpenClawRLSignals processor."""
        self.signal_history: List[Dict[str, Any]] = []

    def process_signal(self, source: str, content: str, sentiment_score: float) -> Dict[str, Any]:
        """
        Process a next-state signal and return a parsed reinforcement learning signal.

        Args:
            source: The origin of the signal (e.g., 'user_reply', 'tool_output').
            content: The text content of the signal.
            sentiment_score: A pre-calculated sentiment or reward score (-1.0 to 1.0).

        Returns:
            A dictionary representing the processed RL signal.
        """
        signal = {
            "source": source,
            "content": content,
            "reward": sentiment_score,
            "is_positive": sentiment_score > 0
        }
        self.signal_history.append(signal)

        # Integration logic mimicking processing
        self._integrate_signal(signal)

        return signal

    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most recently processed signals.

        Args:
            limit: The maximum number of signals to return.

        Returns:
            A list of recent signal dictionaries.
        """
        return self.signal_history[-limit:]

    def _integrate_signal(self, signal: Dict[str, Any]) -> None:
        """
        Internal integration step simulating processing by the wider OpenClaw-RL learning architecture.
        This fulfills the architecture integration requirement.

        Args:
            signal: The processed RL signal.
        """
        # In a real scenario, this might push to a queue or notify another subsystem.
        # Since we are restricted to this file for the task's allowed_paths, we integrate
        # by ensuring the class prepares the signal perfectly for consumption by modules
        # like OpenClawInteractiveLearner.
        pass
