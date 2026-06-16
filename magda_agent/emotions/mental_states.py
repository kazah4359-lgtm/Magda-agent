from typing import Dict, Optional

class MentalState:
    def __init__(self, fear: float = 0.0, desire: float = 0.5, tension: float = 0.0, optimism: float = 0.5, overconfidence: float = 0.0):
        self.fear = fear
        self.desire = desire
        self.tension = tension
        self.optimism = optimism
        self.overconfidence = overconfidence

class MentalStates:
    """
    Tracks and manages cognitive mental states: Fear, Desire, and Tension.
    These states complement the PAD emotional model to represent higher-level cognitive drives.
    """
    def __init__(self):
        self._states: Dict[int, MentalState] = {}

    def _get_state(self, user_id: Optional[int]) -> MentalState:
        u_id = user_id if user_id is not None else -1
        if u_id not in self._states:
            self._states[u_id] = MentalState()
        return self._states[u_id]

    def update_from_action_result(self, success: bool, user_id: Optional[int] = None) -> None:
        """
        Updates mental states based on the success or failure of an agent action.
        """
        state = self._get_state(user_id)
        if success:
            state.fear = self._clamp(state.fear - 0.1)
            state.desire = self._clamp(state.desire + 0.1)
            state.tension = self._clamp(state.tension - 0.2)
            state.optimism = self._clamp(state.optimism + 0.05)
            state.overconfidence = self._clamp(state.overconfidence + 0.05)
        else:
            state.fear = self._clamp(state.fear + 0.1)
            state.tension = self._clamp(state.tension + 0.1)
            state.desire = self._clamp(state.desire - 0.1)
            state.optimism = self._clamp(state.optimism - 0.1)
            state.overconfidence = self._clamp(state.overconfidence - 0.05)

    def apply_bias_modifier(self, optimism_mod: float = 0.0, overconfidence_mod: float = 0.0, user_id: Optional[int] = None) -> None:
        """
        Manually apply modifiers to cognitive biases.
        """
        state = self._get_state(user_id)
        state.optimism = self._clamp(state.optimism + optimism_mod)
        state.overconfidence = self._clamp(state.overconfidence + overconfidence_mod)

    def get_state_label(self, user_id: Optional[int] = None) -> str:
        """
        Maps numerical values to human-readable labels.
        """
        state = self._get_state(user_id)
        if state.desire > 0.7:
            return "Determined"
        if state.fear > 0.5:
            return "Anxious"
        if state.tension > 0.6:
            return "Focused"
        if state.fear < 0.3 and state.tension < 0.3:
            return "Satisfied"
        return "Stable"

    def get_summary(self, user_id: Optional[int] = None) -> str:
        """
        Returns a formatted summary of the current mental states.
        """
        state = self._get_state(user_id)
        label = self.get_state_label(user_id)
        return f"Mental State: {label} (Fear: {state.fear:.2f}, Desire: {state.desire:.2f}, Tension: {state.tension:.2f}, Optimism: {state.optimism:.2f}, Overconfidence: {state.overconfidence:.2f})"

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        return max(min_val, min(max_val, value))
