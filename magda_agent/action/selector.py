from typing import List, Dict, Any, Optional

class BasalGanglia:
    """
    Basal Ganglia (Action Selection) module.
    Responsible for selecting the most appropriate action from a list of possible actions
    based on their assigned priorities.
    """

    def __init__(self, policy_layer: Optional[Any] = None) -> None:
        """
        Initializes the Basal Ganglia module.

        Args:
            policy_layer (Optional[Any]): The policy layer to evaluate actions.
        """
        self.policy_layer = policy_layer

    def select_action(self, actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the action with the highest priority from the provided list,
        filtering out those denied by the policy layer.

        Args:
            actions (List[Dict[str, Any]]): A list of dictionaries representing possible actions.
                                            Each action should have a 'priority' key (int or float),
                                            and an 'action' key (str) representing the tool name.

        Returns:
            Optional[Dict[str, Any]]: The selected action, or None if the list is empty or all actions are denied.
        """
        if not actions:
            return None

        # Sort actions by priority in descending order
        sorted_actions = sorted(actions, key=lambda a: a.get('priority', 0), reverse=True)

        for action_dict in sorted_actions:
            action_name = action_dict.get("action")

            # If there's a policy layer and an action name, evaluate it
            if self.policy_layer and action_name:
                # Extract kwargs if any (assuming they are stored in 'kwargs' key, or pass entire dict except 'priority'/'action')
                kwargs = action_dict.get("kwargs", {})
                allow, explanation = self.policy_layer.evaluate(action_name, **kwargs)

                if not allow:
                    # Keep looking for the next best action
                    continue

            return action_dict

        return None
