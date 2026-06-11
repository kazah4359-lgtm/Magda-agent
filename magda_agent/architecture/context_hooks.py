import logging
from typing import Callable, Dict, List, Any, Optional

class HookRegistry:
    """Registry and manager for context lifecycle hooks."""

    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable]] = {}
        logging.debug("Initialized HookRegistry")

    def register_hook(self, hook_type: str, callback: Callable) -> None:
        """Registers a callback for a specific hook type."""
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._hooks[hook_type].append(callback)
        logging.debug(f"Registered hook '{hook_type}': {callback.__name__}")

    def trigger_hook(self, hook_type: str, *args: Any, **kwargs: Any) -> Any:
        """Triggers all callbacks registered for the hook type."""
        logging.debug(f"Triggering hook '{hook_type}'")
        if hook_type not in self._hooks or not self._hooks[hook_type]:
            # For pipeline-style hooks like before_retrieval, return the first argument
            if args:
                return args[0]
            return None

        result = None
        if args:
            result = args[0]

        for callback in self._hooks[hook_type]:
            if result is not None:
                # pass result as first arg for chaining
                new_args = (result,) + args[1:]
                result = callback(*new_args, **kwargs)
            else:
                result = callback(*args, **kwargs)

        return result
