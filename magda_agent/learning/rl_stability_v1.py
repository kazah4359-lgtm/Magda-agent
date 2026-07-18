import logging
from typing import Dict, List, Optional
from magda_agent.evaluation.longitudinal_metrics import LongitudinalMetricsTracker

class RLSkillWeightStabilityTracker:
    """
    Tracks the stability of skill weights in Procedural Memory.
    Identifies rapid oscillations and logs stability metrics.
    """

    def __init__(self, metrics_tracker: LongitudinalMetricsTracker, window_size: int = 5, oscillation_threshold: int = 3) -> None:
        """
        Initializes the RLSkillWeightStabilityTracker.

        Args:
            metrics_tracker (LongitudinalMetricsTracker): The tracker for longitudinal metrics.
            window_size (int): The number of recent updates to keep per skill.
            oscillation_threshold (int): The number of direction changes to trigger an oscillation warning.
        """
        self.metrics_tracker = metrics_tracker
        self.window_size = window_size
        self.oscillation_threshold = oscillation_threshold
        # Keeps track of recent deltas per skill
        self.skill_deltas: Dict[str, List[float]] = {}
        # Keeps track of the most recent weight per skill
        self.skill_weights: Dict[str, float] = {}

    def record_weight_update(self, skill_id: str, new_weight: float) -> bool:
        """
        Records a weight update, tracks deltas, and checks for rapid oscillations.

        Args:
            skill_id (str): The identifier for the skill.
            new_weight (float): The newly updated weight.

        Returns:
            bool: True if a rapid oscillation is detected, False otherwise.
        """
        if skill_id not in self.skill_weights:
            self.skill_weights[skill_id] = new_weight
            self.skill_deltas[skill_id] = []
            return False

        old_weight = self.skill_weights[skill_id]
        delta = new_weight - old_weight
        self.skill_weights[skill_id] = new_weight

        # We keep track of the change (delta)
        self.skill_deltas[skill_id].append(delta)

        if len(self.skill_deltas[skill_id]) > self.window_size:
            self.skill_deltas[skill_id].pop(0)

        is_oscillating = self._detect_oscillation(skill_id)

        # Log metric: 0.0 for oscillating (unstable), 1.0 for stable
        stability_score = 0.0 if is_oscillating else 1.0
        self.metrics_tracker.record_metric(f"rl_stability_{skill_id}", stability_score)

        if is_oscillating:
            logging.warning(f"Rapid oscillation detected for skill weight: {skill_id}")

        return is_oscillating

    def _detect_oscillation(self, skill_id: str) -> bool:
        """
        Detects if there are frequent direction changes in recent deltas.

        Args:
            skill_id (str): The identifier for the skill.

        Returns:
            bool: True if direction changes meet or exceed the threshold.
        """
        deltas = self.skill_deltas.get(skill_id, [])
        if len(deltas) < self.oscillation_threshold:
            return False

        direction_changes = 0
        for i in range(1, len(deltas)):
            # If the product of adjacent deltas is negative, the direction changed
            if deltas[i-1] * deltas[i] < 0:
                direction_changes += 1

        return direction_changes >= self.oscillation_threshold
