"""
Subagent Spawning and Context Compression.

This module provides the SubagentSpawner class which enables dynamic
subagent spawning for parallel execution with compressed context passing
to optimize token usage, inspired by the Claude Agent SDK.
"""

from typing import List, Dict, Any

class SubagentSpawner:
    """
    Manages the dynamic spawning of subagents with isolated context boundaries.
    """

    def __init__(self, max_context_tokens: int = 4000):
        """
        Initialize the SubagentSpawner.

        Args:
            max_context_tokens: Maximum allowed token threshold for context.
        """
        self.max_context_tokens = max_context_tokens

    def compress_context(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compress the context to optimize token usage before passing it to a subagent.

        Currently implements a naive strategy: if context has more than 5 messages,
        it keeps the first one (usually system prompt) and the last 4.

        Args:
            context: The full context, typically a list of message dicts.

        Returns:
            The compressed context.
        """
        if not context:
            return []

        # Naive compression logic: keep system prompt and last N messages
        if len(context) > 5:  # naive usage ignores self.max_context_tokens for now
            compressed = [context[0]] + context[-4:]
            return compressed

        return context

    async def spawn_subagent(
        self,
        task_description: str,
        full_context: List[Dict[str, Any]],
        agent_executor: Any
    ) -> Any:
        """
        Spawn a subagent to execute a specific task with compressed context.

        Args:
            task_description: The task the subagent should perform.
            full_context: The full conversation or execution context.
            agent_executor: An async callable or object with an `execute` method
                            that runs the subagent.

        Returns:
            The result of the subagent's execution.
        """
        compressed_context = self.compress_context(full_context)

        # We append the specific task to the compressed context
        execution_context = compressed_context.copy()
        execution_context.append({
            "role": "user",
            "content": f"Task: {task_description}"
        })

        if hasattr(agent_executor, "execute") and callable(agent_executor.execute):
            return await agent_executor.execute(execution_context)
        elif callable(agent_executor):
            return await agent_executor(execution_context)
        else:
            raise TypeError("agent_executor must be callable or have an execute method")
