import asyncio
import logging
from typing import List, Dict, Any

from magda_agent.llm_client import LLMClient
from magda_agent.memory.context_engine import ContextEngine
from magda_agent.agents.spawner import SubagentSpawner

class ContextAwareTeamSpawner:
    """
    Spawns SubAgents with context dynamically managed by the OpenClaw ContextEngine.
    """
    def __init__(self, llm: LLMClient, context_engine: ContextEngine) -> None:
        """
        Initializes the ContextAwareTeamSpawner.

        Args:
            llm: The Language Model client to be used by spawned SubAgents.
            context_engine: The ContextEngine to enrich task context.
        """
        self.llm = llm
        self.context_engine = context_engine
        self.spawner = SubagentSpawner(llm=llm)

    async def spawn_and_execute(self, tasks: List[Dict[str, Any]], base_context: str, user_id: int, use_isolation: bool = True) -> List[str]:
        """
        Spawns subagents dynamically to execute multiple tasks concurrently with enriched context.

        Args:
            tasks: A list of task dictionaries containing a 'description' field.
            base_context: The shared base context to start with.
            user_id: The ID of the user requesting the task execution.
            use_isolation: Whether to use Git Worktree isolation for each subagent.

        Returns:
            A list of execution results corresponding to the tasks.
        """
        logging.info(f"ContextAwareTeamSpawner processing {len(tasks)} tasks.")

        async def execute_task(task: Dict[str, Any]) -> str:
            """Executes a single task within the isolated context."""
            query = task.get('description', '')

            # Retrieve context items specifically for this task query
            context_items = self.context_engine.retrieve_context(
                query=query,
                user_id=user_id,
                base_retrieval_func=lambda q, uid: [base_context]
            )

            # Assemble context into a single string
            assembled_context = await self.context_engine.assemble(
                context_items=context_items,
                metadata={'user_id': user_id, 'task_query': query}
            )

            # Execute a single task via SubagentSpawner with the enriched context
            results = await self.spawner.spawn_and_execute(
                tasks=[task],
                base_context=assembled_context,
                use_isolation=use_isolation
            )
            return results[0] if results else ""

        # Execute all tasks concurrently
        results = await asyncio.gather(*(execute_task(t) for t in tasks))
        return list(results)
