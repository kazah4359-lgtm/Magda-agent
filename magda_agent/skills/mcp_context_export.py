import json
from typing import Dict, Any, List, Optional
import inspect

class MCPContextExportTool:
    """
    Exports the current state of Context Engine or Working Memory as an MCP-compatible tool.
    This allows external visualization applications like OpenClaw Canvas to retrieve the context state.
    """
    def __init__(self, memory_source: Any) -> None:
        """
        Initializes the export tool.

        Args:
            memory_source: An instance of WorkingMemory, ContextEngine, or similar
                           object that has a `get_entries` or `get_all_entries` method.
        """
        self.memory_source = memory_source
        self.name = "export_context_state"
        self.description = "Export the current working memory or context engine state."

    def _get_json_schema(self) -> Dict[str, Any]:
        """
        Returns the MCP tool input schema.
        """
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "Optional user ID to filter the context. If omitted, exports all available context."
                }
            },
            "required": []
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists this tool in MCP format.
        """
        return [{
            "name": self.name,
            "description": self.description,
            "inputSchema": self._get_json_schema()
        }]

    def _extract_entries(self, user_id: Optional[int] = None) -> List[Any]:
        """
        Extracts entries from the memory source depending on its interface.
        """
        if hasattr(self.memory_source, "get_entries"):
            if user_id is not None:
                return self.memory_source.get_entries(user_id=user_id)
            elif hasattr(self.memory_source, "get_all_entries"):
                return self.memory_source.get_all_entries()
            else:
                return self.memory_source.get_entries()
        return []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes the tool via the MCP protocol format.
        """
        if name != self.name:
            return {
                "content": [{"type": "text", "text": f"Error: Tool '{name}' not found."}],
                "isError": True
            }

        try:
            user_id = arguments.get("user_id")
            entries = self._extract_entries(user_id)

            # Serialize entries to dicts for JSON
            serialized_entries = []
            for entry in entries:
                entry_dict = {}
                if hasattr(entry, "id"):
                    entry_dict["id"] = entry.id
                if hasattr(entry, "content"):
                    entry_dict["content"] = entry.content
                if hasattr(entry, "importance"):
                    entry_dict["importance"] = entry.importance
                if hasattr(entry, "timestamp"):
                    entry_dict["timestamp"] = entry.timestamp
                if hasattr(entry, "tags"):
                    entry_dict["tags"] = entry.tags
                if hasattr(entry, "user_id"):
                    entry_dict["user_id"] = entry.user_id

                serialized_entries.append(entry_dict)

            json_payload = json.dumps(serialized_entries)

            return {
                "content": [{"type": "text", "text": json_payload}],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error executing tool {name}: {e}"}],
                "isError": True
            }

    async def call_tool_async(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of call_tool.
        """
        return self.call_tool(name, arguments)
