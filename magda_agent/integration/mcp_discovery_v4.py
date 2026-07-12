import asyncio
import os
import importlib.util
from typing import List, Dict, Any, Callable, Optional, Set
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class MCPTool(BaseModel):
    name: str
    description: str
    callable_func: Callable

class MCPDynamicDiscoveryV4:
    """
    Auto-discovery mechanism that scans a configured tools directory and automatically registers new MCP tools without requiring restart.
    """
    def __init__(self, tools_dir: str) -> None:
        """
        Initializes the discovery mechanism with the tools directory.
        """
        self.tools_dir = tools_dir
        self.tools: Dict[str, MCPTool] = {}
        self._last_mtime: Dict[str, float] = {}
        # Keep track of which tools came from which file to allow cleanup
        self._file_to_tools: Dict[str, Set[str]] = {}

    async def scan_and_register(self) -> None:
        """Scans the tools directory and registers new or modified tools, and removes deleted ones."""
        if not os.path.exists(self.tools_dir):
            logger.warning(f"Tools directory {self.tools_dir} does not exist.")
            return

        current_files = set()

        for filename in os.listdir(self.tools_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                filepath = os.path.join(self.tools_dir, filename)
                current_files.add(filepath)
                mtime = os.path.getmtime(filepath)

                if filepath not in self._last_mtime or self._last_mtime[filepath] < mtime:
                    try:
                        module_name = filename[:-3]
                        spec = importlib.util.spec_from_file_location(module_name, filepath)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            # Cleanup old tools from this file if it was modified
                            if filepath in self._file_to_tools:
                                for old_tool_name in self._file_to_tools[filepath]:
                                    if old_tool_name in self.tools:
                                        del self.tools[old_tool_name]

                            self._file_to_tools[filepath] = set()

                            # Looking for register_tool function or TOOLS list
                            if hasattr(module, 'register_tool'):
                                tool = module.register_tool()
                                if isinstance(tool, MCPTool):
                                    self.tools[tool.name] = tool
                                    self._file_to_tools[filepath].add(tool.name)
                                    logger.info(f"Registered tool {tool.name} from {filename}")
                            elif hasattr(module, 'TOOLS'):
                                for tool in module.TOOLS:
                                     if isinstance(tool, MCPTool):
                                         self.tools[tool.name] = tool
                                         self._file_to_tools[filepath].add(tool.name)
                                logger.info(f"Registered tools from {filename}")
                    except Exception as e:
                        logger.error(f"Error loading tool from {filename}: {e}")
                    finally:
                         # Always update mtime even if it fails or has no tools, to prevent continuous reloading
                         self._last_mtime[filepath] = mtime

        # Cleanup deleted files
        deleted_files = set(self._last_mtime.keys()) - current_files
        for filepath in deleted_files:
            if filepath in self._file_to_tools:
                 for old_tool_name in self._file_to_tools[filepath]:
                     if old_tool_name in self.tools:
                         del self.tools[old_tool_name]
                         logger.info(f"Removed tool {old_tool_name} from deleted file {filepath}")
                 del self._file_to_tools[filepath]
            del self._last_mtime[filepath]

    def get_registered_tools(self) -> List[MCPTool]:
        """Returns a list of all registered tools."""
        return list(self.tools.values())
