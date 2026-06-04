from typing import List, Dict, Any, Optional

class BasalGanglia:
    """
    Basal Ganglia (Action Selection) module.
    Responsible for selecting the most appropriate action from a list of possible actions
    based on their assigned priorities.
    """

    def __init__(self) -> None:
        """
        Initializes the Basal Ganglia module.
        """
        pass

    def select_action(self, actions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the action with the highest priority from the provided list.

        Args:
            actions (List[Dict[str, Any]]): A list of dictionaries representing possible actions.
                                            Each action should have a 'priority' key (int or float).

        Returns:
            Optional[Dict[str, Any]]: The selected action, or None if the list is empty.
        """
        if not actions:
            return None

        # Sort actions by priority in descending order and return the first one
        # If 'priority' is missing, default to 0
        selected = max(actions, key=lambda a: a.get('priority', 0))
        return selected
