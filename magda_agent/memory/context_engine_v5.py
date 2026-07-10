import logging
from typing import Callable, Any, Optional, List, Dict
import inspect
from magda_agent.architecture.context_hooks_v5 import HookRegistry
from magda_agent.memory.context_engine import ContextEngine, ContextPlugin

class DynamicHookRegistry(HookRegistry):
    """
    Extends HookRegistry with the ability to unregister hooks dynamically.
    """
    def unregister_hook(self, hook_type: str, callback: Callable) -> None:
        """Removes a previously registered callback for a specific hook type."""
        if hook_type in self._hooks:
            # We want to remove the specific callback.
            try:
                self._hooks[hook_type].remove(callback)
                callback_name = getattr(callback, '__name__', str(callback))
                logging.debug(f"Unregistered hook '{hook_type}': {callback_name}")
            except ValueError:
                # Callback was not in the list for this hook_type
                logging.warning(f"Attempted to unregister a hook '{hook_type}' that was not registered.")
        else:
            logging.warning(f"Attempted to unregister from an unknown hook type '{hook_type}'.")


class ContextEngineV5(ContextEngine):
    """
    ContextEngineV5 manages context dynamically using a plugin architecture
    with lifecycle hooks, supporting dynamic addition and removal of plugins.
    """
    def __init__(self, plugins: Optional[List[ContextPlugin]] = None, llm: Optional[Any] = None) -> None:
        # Initialize base attributes
        self._plugins: List[ContextPlugin] = []
        self.llm = llm

        # Override hook_registry with our dynamic version
        self.hook_registry = DynamicHookRegistry()

        if plugins:
            for plugin in plugins:
                self.register_plugin(plugin)

    def unregister_plugin(self, plugin: ContextPlugin) -> None:
        """Unregisters an existing plugin from the context engine."""
        if plugin not in self._plugins:
            logging.warning(f"Attempted to unregister plugin {plugin.__class__.__name__} which is not registered.")
            return

        # Remove from plugin list
        self._plugins.remove(plugin)

        # Unregister Sync hooks
        if hasattr(plugin, 'before_retrieval'):
            self.hook_registry.unregister_hook('before_retrieval', plugin.before_retrieval)
        if hasattr(plugin, 'after_retrieval'):
            self.hook_registry.unregister_hook('after_retrieval', plugin.after_retrieval)
        if hasattr(plugin, 'on_context_update'):
            self.hook_registry.unregister_hook('on_context_update', plugin.on_context_update)
        if hasattr(plugin, 'before_write'):
            self.hook_registry.unregister_hook('before_write', plugin.before_write)
        if hasattr(plugin, 'after_write'):
            self.hook_registry.unregister_hook('after_write', plugin.after_write)

        # Unregister Async and common lifecycle hooks
        if hasattr(plugin, 'bootstrap'):
            self.hook_registry.unregister_hook('bootstrap', plugin.bootstrap)
        if hasattr(plugin, 'ingest'):
            self.hook_registry.unregister_hook('ingest', plugin.ingest)
        if hasattr(plugin, 'assemble'):
            self.hook_registry.unregister_hook('assemble', plugin.assemble)
        if hasattr(plugin, 'compact'):
            self.hook_registry.unregister_hook('compact', plugin.compact)

        logging.debug(f"Unregistered plugin: {plugin.__class__.__name__}")
