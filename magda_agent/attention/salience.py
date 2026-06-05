"""
Salience Network Module

This module implements the Salience Network, responsible for determining the urgency
and importance of events based on heuristics like urgency, test failures, and security risks.
"""

from typing import Tuple, Dict, Any


class SalienceNetwork:
    """
    Evaluates events to determine their salience score.
    Scores events based on multiple factors:
    - urgency
    - active task relevance
    - CI/test signals
    - uncertainty
    - novelty
    - security risk
    """

    def __init__(self):
        pass

    def score_event(self, event: Dict[str, Any]) -> Tuple[float, str]:
        """
        Scores an event and returns a salience score between 0.0 and 1.0,
        along with an explanation string.

        Args:
            event: A dictionary representing the event. Expected keys might include:
                   'content' (str), 'type' (str), 'is_ci_failure' (bool),
                   'is_security_risk' (bool), 'urgency' (float).

        Returns:
            A tuple of (score, explanation).
        """
        score = 0.1  # Base salience for any event
        reasons = ["base score 0.1"]

        content = str(event.get("content", "")).lower()

        # Check for noise or low-value input
        if not content or len(content.strip()) < 2:
            return 0.0, "noisy or empty input"

        # Security risk
        if event.get("is_security_risk", False) or "security" in content or "vulnerability" in content or "exploit" in content:
            score += 0.8
            reasons.append("security risk detected (+0.8)")

        # CI failure
        if event.get("is_ci_failure", False) or "ci failed" in content or "test failed" in content or "traceback" in content:
            score += 0.6
            reasons.append("CI/test failure detected (+0.6)")

        # Urgency keywords
        if "urgent" in content or "asap" in content or "emergency" in content:
            score += 0.4
            reasons.append("urgency keywords detected (+0.4)")

        # Uncertainty
        if event.get("uncertainty", 0.0) > 0.5:
             score += 0.2
             reasons.append("high uncertainty (+0.2)")

        # Novelty
        if event.get("novelty", 0.0) > 0.5:
             score += 0.2
             reasons.append("high novelty (+0.2)")

        # Explicit urgency score
        urgency = event.get("urgency", 0.0)
        if urgency > 0:
             score += urgency * 0.5
             reasons.append(f"explicit urgency ({urgency} * 0.5 = +{urgency*0.5})")

        # Clamp score between 0.0 and 1.0
        final_score = max(0.0, min(1.0, score))
        explanation = ", ".join(reasons)

        return final_score, explanation
