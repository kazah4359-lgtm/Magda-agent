from typing import Dict, Any, List, Callable, Optional
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.skills.registry import SkillRegistry
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory

class MCPContextExporter:
    """
    Exports Magda's Context Engine memory states as MCP tools.
    """
    def __init__(self, context_engine: ContextEngine, registry: Optional[SkillRegistry] = None, episodic_memory: Optional[EpisodicMemory] = None, semantic_memory: Optional[SemanticMemory] = None) -> None:
        """Initialize MCPContextExporter with ContextEngine and optional dependencies."""
        self.context_engine = context_engine
        self.registry = registry
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory

        # If a registry is provided, register the tools
        if self.registry:
            self._register_tools()

    def _register_tools(self) -> None:
        """Registers memory context tools to the skill registry."""
        if not self.registry:
            return

        self.registry.register_skill(
            name="get_context",
            func=self.get_context,
            description="Retrieves the current context from the context engine using both episodic and semantic memories."
        )

    def get_context(self, query: str, user_id: int) -> List[Any]:
        """
        Retrieves the current context from the context engine.
        """
        def base_retrieval(q: str, uid: int) -> List[Any]:
            """Retrieve from actual episodic and semantic memory."""
            context_items = []
            if self.episodic_memory:
                context_items.extend(self.episodic_memory.recall_events(query=q, top_k=5, user_id=uid))
            if self.semantic_memory:
                context_items.extend(self.semantic_memory.recall_facts(query=q, top_k=5, user_id=uid))
            return context_items

        return self.context_engine.retrieve_context(query, user_id, base_retrieval_func=base_retrieval)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists exported tools for the context.
        """
        return [
            {
                "name": "get_context",
                "description": "Retrieves the current context from the context engine using both episodic and semantic memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "integer"}
                    },
                    "required": ["query", "user_id"]
                }
            }
        ]
