import asyncio
import logging
import uuid
from typing import List, Dict, Any

from magda_agent.llm_client import LLMClient
from magda_agent.agents.isolation_v3 import AgentWorktreeIsolationV3
from magda_agent.memory.subagent_compression import SubagentContextCompressor

class SubagentSpawnerV4:
    """
    SubagentSpawnerV4 dynamically spawns isolated contexts for concurrent subagent tasks.
    Leverages AgentWorktreeIsolationV3 for enhanced state isolation without leaking,
    and SubagentContextCompressor for optimizing context token usage.
    Inspired by Claude Agent SDK: Agent Teams and Subagent spawning.
    """
    def __init__(self, llm: LLMClient, base_dir: str = "/tmp/magda_team_worktrees_v4") -> None:
        """
        Initializes the SubagentSpawnerV4.

        Args:
            llm: The Language Model client to be used by spawned contexts and compression.
            base_dir: The base directory where git worktrees will be created.
        """
        self.llm = llm
        self.isolation_manager = AgentWorktreeIsolationV3(base_dir=base_dir)
        self.context_compressor = SubagentContextCompressor(llm=llm)

    async def spawn_and_execute(self, tasks: List[Dict[str, Any]], base_context: str) -> List[str]:
        """
        Spawns subagents dynamically to execute multiple tasks concurrently.
        Each task receives its own compressed and isolated git worktree context.

        Args:
            tasks: A list of task dictionaries containing a 'description' and optional 'system_prompt'.
            base_context: The shared context passed to all subagents, which will be compressed.

        Returns:
            A list of execution results corresponding to the tasks.
        """
        logging.info(f"SubagentSpawnerV4 spawning {len(tasks)} isolated subagents for parallel execution.")

        async def run_subagent(task_spec: Dict[str, Any]) -> str:
            agent_id = f"subagent_v4_{uuid.uuid4().hex[:8]}"
            task_description = task_spec.get('description', 'Unknown task')
            system_prompt = task_spec.get('system_prompt', "You are an isolated Sub-Agent executing a specific task.")

            try:
                # 1. Compress Base Context
                compressed_context = await self.context_compressor.compress_context(base_context)

                # 2. Setup Isolation
                worktree_path = await self.isolation_manager.setup_isolation(agent_id=agent_id)
                logging.info(f"Agent {agent_id} isolated at {worktree_path}")

                # 3. Build Context
                augmented_context = f"{compressed_context}\n\nIsolated Git Worktree Path: {worktree_path}\n\nAssigned Task:\n{task_description}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": augmented_context}
                ]

                # 4. Execute Task (mocked LLM for simplicity, assumes SubAgent logic equivalent)
                result = await self.llm.chat_completion(messages)
                return result

            except Exception as e:
                logging.error(f"Error executing SubAgent task {agent_id}: {e}")
                return f"Error executing SubAgent task: {e}"

            finally:
                # 5. Teardown Isolation
                try:
                    await self.isolation_manager.teardown_isolation(agent_id=agent_id)
                    logging.info(f"Agent {agent_id} isolation teardown complete.")
                except Exception as cleanup_error:
                    logging.error(f"Failed to cleanup isolation for {agent_id}: {cleanup_error}")

        results = await asyncio.gather(*(run_subagent(task) for task in tasks))
        return list(results)
