import logging
from typing import Dict, List, Optional

class ConversationStep:
    """
    Represents a single step in a conversational trajectory.
    """
    def __init__(self, skill_id: str, state_context: str, q_value: float = 0.0) -> None:
        """
        Initializes a ConversationStep.

        Args:
            skill_id (str): The ID of the skill used in this step.
            state_context (str): The context of the state/action.
            q_value (float): The current Q-value associated with this step/skill.
        """
        self.skill_id = skill_id
        self.state_context = state_context
        self.q_value = q_value

class TrajectoryBuffer:
    """
    Buffers conversation steps for a user session.
    """
    def __init__(self, max_size: int = 20) -> None:
        """
        Initializes a TrajectoryBuffer.

        Args:
            max_size (int): The maximum number of steps to retain.
        """
        self.steps: List[ConversationStep] = []
        self.max_size = max_size

    def add_step(self, step: ConversationStep) -> None:
        """
        Adds a conversation step to the buffer, evicting the oldest if limit is reached.

        Args:
            step (ConversationStep): The step to add.
        """
        self.steps.append(step)
        if len(self.steps) > self.max_size:
            self.steps.pop(0)

    def clear(self) -> None:
        """
        Clears the buffer.
        """
        self.steps.clear()

class OpenClawRLTrajectoryRolloutV6:
    """
    Handles buffering of multi-turn trajectories and backpropagation of delayed rewards.
    Inspired by OpenClaw-RL (June 2026).
    """
    def __init__(self, initial_q_values: Optional[Dict[str, float]] = None) -> None:
        """
        Initializes the OpenClawRLTrajectoryRolloutV6 learner.

        Args:
            initial_q_values (Optional[Dict[str, float]]): Optional initial Q-values for skills.
        """
        self.q_table: Dict[str, float] = initial_q_values or {}
        self.buffers: Dict[str, TrajectoryBuffer] = {}
        logging.info("Initialized OpenClawRLTrajectoryRolloutV6")

    def get_buffer(self, user_id: str) -> TrajectoryBuffer:
        """
        Retrieves or creates a TrajectoryBuffer for a specific user.

        Args:
            user_id (str): The unique identifier of the user.

        Returns:
            TrajectoryBuffer: The user's trajectory buffer.
        """
        user_key = str(user_id)
        if user_key not in self.buffers:
            self.buffers[user_key] = TrajectoryBuffer()
        return self.buffers[user_key]

    def record_step(self, user_id: str, skill_id: str, state_context: str) -> None:
        """
        Records a conversation step for a specific user, persisting it across turns.

        Args:
            user_id (str): The unique identifier of the user.
            skill_id (str): The ID of the skill used.
            state_context (str): The context of the state/action.
        """
        buffer = self.get_buffer(user_id)
        current_q = self.q_table.get(skill_id, 0.0)
        step = ConversationStep(skill_id, state_context, current_q)
        buffer.add_step(step)
        logging.info(f"Recorded step for user '{user_id}': skill='{skill_id}', Q-value={current_q:.4f}")

    def process_delayed_reward(
        self,
        user_id: str,
        final_reward: float,
        discount_factor: float = 0.9,
        learning_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Backpropagates a delayed reward to all previous steps in the buffer.
        The reward decays exponentially based on the distance from the last step (index).

        Args:
            user_id (str): The unique identifier of the user.
            final_reward (float): The delayed reward signal.
            discount_factor (float): The discount factor (gamma) for delayed rewards.
            learning_rate (float): The learning rate (alpha) for Q-value updates.

        Returns:
            Dict[str, float]: The updated Q-value dictionary for skills updated.
        """
        buffer = self.get_buffer(user_id)
        steps = buffer.steps
        if not steps:
            logging.warning(f"No steps buffered for user '{user_id}'. Cannot backpropagate reward.")
            return {}

        updated_values: Dict[str, float] = {}
        n = len(steps)
        for idx in range(n):
            step = steps[idx]
            # Distance from the end (latest step)
            distance_from_end = n - 1 - idx
            discounted_reward = final_reward * (discount_factor ** distance_from_end)

            # Update the Q-value: Q = Q + alpha * (discounted_reward - Q)
            current_q = self.q_table.get(step.skill_id, 0.0)
            td_error = discounted_reward - current_q
            new_q = current_q + learning_rate * td_error

            self.q_table[step.skill_id] = new_q
            updated_values[step.skill_id] = new_q
            step.q_value = new_q

        logging.info(f"Backpropagated delayed reward {final_reward} to {n} steps for user '{user_id}'.")
        # Clear the buffer after rollout to prevent double reward application
        buffer.clear()
        return updated_values

    def get_q_value(self, skill_id: str) -> float:
        """
        Retrieves the Q-value for a given skill.

        Args:
            skill_id (str): The ID of the skill.

        Returns:
            float: The current Q-value.
        """
        return self.q_table.get(skill_id, 0.0)
