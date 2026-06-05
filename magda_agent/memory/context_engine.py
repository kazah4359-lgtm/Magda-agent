import logging
from typing import List, Dict, Any, Optional

class ContextPlugin:
    """
    Base class for Context Engine plugins.
    Defines lifecycle hooks for manipulating memory and context.
    """
    async def pre_process(self, input_data: Any) -> Any:
        """Called before the context is processed or added to memory."""
        return input_data

    async def compress(self, context_data: Any) -> Any:
        """Called to compress or summarize the context data when limits are reached."""
        return context_data

    async def post_process(self, output_data: Any) -> Any:
        """Called after the context is retrieved or final assembly occurs."""
        return output_data

class ContextEngine:
    """
    Context Engine that manages plugins and coordinates memory modules dynamically.
    Dispatches lifecycle events (pre_process, compress, post_process) to registered plugins.
    """
    def __init__(self):
        self.plugins: List[ContextPlugin] = []
        self._logger = logging.getLogger(__name__)

    def register_plugin(self, plugin: ContextPlugin) -> None:
        """Registers a new context plugin."""
        self.plugins.append(plugin)

    async def dispatch_pre_process(self, input_data: Any) -> Any:
        """Dispatch pre_process hook across all plugins."""
        current_data = input_data
        for plugin in self.plugins:
            try:
                current_data = await plugin.pre_process(current_data)
            except Exception as e:
                self._logger.error(f"Plugin {plugin.__class__.__name__} failed on pre_process: {e}")
        return current_data

    async def dispatch_compress(self, context_data: Any) -> Any:
        """Dispatch compress hook across all plugins."""
        current_data = context_data
        for plugin in self.plugins:
            try:
                current_data = await plugin.compress(current_data)
            except Exception as e:
                self._logger.error(f"Plugin {plugin.__class__.__name__} failed on compress: {e}")
        return current_data

    async def dispatch_post_process(self, output_data: Any) -> Any:
        """Dispatch post_process hook across all plugins."""
        current_data = output_data
        for plugin in self.plugins:
            try:
                current_data = await plugin.post_process(current_data)
            except Exception as e:
                self._logger.error(f"Plugin {plugin.__class__.__name__} failed on post_process: {e}")
        return current_data
