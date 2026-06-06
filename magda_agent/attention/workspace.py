"""
Global Workspace Module

This module implements the Global Workspace, a central cognitive loop
component where multiple candidate events compete for focus based on
their salience score before memory retrieval, planning, and action selection.
"""

from typing import Dict, Any, List, Optional, Callable
from magda_agent.attention.salience import SalienceNetwork


class GlobalWorkspace:
    """
    Manages candidate events and selects the focal event based on salience.
    """

    def __init__(self, salience_network: SalienceNetwork):
        self.salience_network = salience_network
        self.candidates: List[Dict[str, Any]] = []
        self.suppressed_candidates: List[Dict[str, Any]] = []
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []


    def register_listener(self, listener_callable: Callable[[Dict[str, Any]], None]) -> None:
        """
        Registers a callback function to receive focus events.

        Args:
            listener_callable: A callable that accepts a single event dictionary.
        """
        self.listeners.append(listener_callable)

    def add_candidate(self, event: Dict[str, Any]) -> None:
        """
        Adds a new candidate event to the workspace.

        Args:
            event: A dictionary representing the event.
        """
        self.candidates.append(event)

    def select_focus(self) -> Optional[Dict[str, Any]]:
        """
        Scores all candidates using the SalienceNetwork and selects the one
        with the highest score to be the focal event.
        The remaining candidates are moved to suppressed_candidates.

        Returns:
            The focused event dictionary, or None if there are no candidates.
        """
        if not self.candidates:
            return None

        scored_candidates = []
        for event in self.candidates:
            score, explanation = self.salience_network.score_event(event)
            # Annotate event with score and explanation
            event["_salience_score"] = score
            event["_salience_explanation"] = explanation
            scored_candidates.append((score, event))

        # Sort candidates by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select the highest scoring candidate
        focus_event = scored_candidates[0][1]

        # Move the rest to suppressed candidates
        self.suppressed_candidates = [event for _, event in scored_candidates[1:]]

        # Clear candidates for the next cycle
        self.candidates = []

        # Broadcast the focused event to all listeners
        for listener in self.listeners:
            listener(focus_event)

        return focus_event


    def get_suppressed_candidates(self) -> List[Dict[str, Any]]:
        """
        Returns the list of suppressed candidate events from the last focus selection.

        Returns:
            A list of event dictionaries.
        """
        return self.suppressed_candidates

    def clear(self) -> None:
        """
        Clears all candidates and suppressed candidates.
        """
        self.candidates = []
        self.suppressed_candidates = []

    def process_interruption(self, new_event: Dict[str, Any], planner: Any) -> bool:
        """
        Evaluates a new event for interruption. If its priority exceeds the current
        plan's risk/priority, pauses the plan and makes the new event a candidate.

        Args:
            new_event: The new incoming event.
            planner: The Prefrontal Cortex (Planner) instance.

        Returns:
            True if an interruption occurred, False otherwise.
        """
        if not planner.current_plan and not planner.completed_steps:
            # No active plan to interrupt, just add as normal candidate
            self.add_candidate(new_event)
            return False

        current_risk = planner.current_risk

        if self.salience_network.evaluate_interrupt(new_event, current_risk):
            planner.pause_current_plan()
            self.add_candidate(new_event)
            return True
        else:
            self.add_candidate(new_event)
            return False
