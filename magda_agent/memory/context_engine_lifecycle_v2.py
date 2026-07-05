import logging
from typing import Dict, Any, Optional
from magda_agent.memory.context_engine_v1 import ContextEngineV1, ContextPluginV1

class ContextPluginLifecycleV2(ContextPluginV1):
    """
    Protocol extending ContextPluginV1 to include lifecycle hooks
    for dynamically loading and unloading plugins.
    """

    async def on_load(self, config: Dict[str, Any]) -> None:
        """Called when the plugin is dynamically loaded."""
        pass

    async def on_unload(self) -> None:
        """Called when the plugin is dynamically unloaded."""
        pass

class ContextEngineLifecycleV2(ContextEngineV1):
    """
    Extends ContextEngineV1 to support dynamic loading and unloading
    of plugins at runtime with lifecycle hooks.
    """

    async def load_plugin(self, plugin: ContextPluginLifecycleV2, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Dynamically loads a plugin, registers its hooks, and calls its on_load hook.
        """
        if config is None:
            config = {}

        self.register_plugin(plugin)

        if hasattr(plugin, 'on_load'):
            await plugin.on_load(config)

        logging.info(f"Dynamically loaded plugin: {plugin.__class__.__name__}")

    async def unload_plugin(self, plugin: ContextPluginLifecycleV2) -> None:
        """
        Dynamically unloads a plugin, calls its on_unload hook, and unregisters its hooks.
        """
        if plugin not in self._plugins:
            logging.warning(f"Plugin {plugin.__class__.__name__} is not registered.")
            return

        if hasattr(plugin, 'on_unload'):
            await plugin.on_unload()

        self._plugins.remove(plugin)

        # Remove plugin's hooks from the registry
        for hook_type, callbacks in self.hook_registry._hooks.items():
            # We need to iterate carefully to remove the correct bound methods
            # related to this plugin instance.
            hooks_to_remove = []
            for cb in callbacks:
                # Check if the callback is a bound method of the plugin instance
                if hasattr(cb, '__self__') and cb.__self__ is plugin:
                    hooks_to_remove.append(cb)

            for cb in hooks_to_remove:
                callbacks.remove(cb)

        logging.info(f"Dynamically unloaded plugin: {plugin.__class__.__name__}")
