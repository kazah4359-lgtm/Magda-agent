import logging
from typing import Optional, List, Dict, Any
from magda_agent.memory.episodic import EpisodicMemory
from magda_agent.memory.semantic import SemanticMemory
from magda_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

class SemanticCompactionV4:
    """
    OpenClaw Semantic Memory Compaction V4
    Inspired by OpenClaw trends: Implement semantic memory compaction that runs periodically via cron, clustering related events.
    """
    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        llm_client: LLMClient,
        batch_size: int = 10,
        cluster_prompt: Optional[str] = None
    ):
        """
        Initializes the SemanticCompactionV4.

        Args:
            episodic_memory: The episodic memory instance to read events from.
            semantic_memory: The semantic memory instance to store clustered facts.
            llm_client: The LLM client to use for clustering/summarization.
            batch_size: Minimum number of events required to trigger compaction.
            cluster_prompt: Optional custom prompt template for clustering.
        """
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.cluster_prompt = cluster_prompt or (
            "Analyze the following episodic memory events and extract the core semantic facts. "
            "Cluster related events and provide a concise summary of the key facts learned. "
            "Return the facts as a simple text list, one fact per line.\n\nEvents:\n{events}"
        )

    async def run_compaction(self) -> None:
        """
        Fetches non-decayed events, clusters them using LLM, stores as semantic facts,
        and decays the original events.
        """
        logger.info("Starting Semantic Memory Compaction V4")

        events = self.episodic_memory.get_all_events(include_decayed=False, limit=self.batch_size * 2)
        if not events or len(events) < self.batch_size:
            logger.info(f"Not enough events for compaction. Found {len(events)}, require {self.batch_size}.")
            return

        events_to_compact = events[:self.batch_size]
        event_texts = [f"- {e['text']}" for e in events_to_compact]
        events_str = "\n".join(event_texts)

        prompt = self.cluster_prompt.format(events=events_str)

        logger.info(f"Clustering {len(events_to_compact)} events.")

        try:
            # We use chat_completion directly or generate?
            # guidelines say we should use generate or chat_completion? LLMClient has both, both async.
            response = await self.llm_client.generate(prompt)

            if response.startswith("Error:"):
                logger.error(f"LLM Error during compaction: {response}")
                return

            # Store the facts
            facts = [fact.strip("- *").strip() for fact in response.split("\n") if fact.strip()]
            for fact in facts:
                if fact:
                    self.semantic_memory.store_fact(fact, metadata={"source": "compaction_v4"})
                    logger.debug(f"Stored compacted fact: {fact}")

            # Decay original events
            for event in events_to_compact:
                self.episodic_memory.decay_event(event["id"])

            logger.info("Semantic compaction completed successfully.")

        except Exception as e:
            logger.error(f"Error during semantic compaction: {e}", exc_info=True)


def register_compaction_cron(
    scheduler,
    episodic_memory: EpisodicMemory,
    semantic_memory: SemanticMemory,
    llm_client: LLMClient,
    cron_expr: str = "0 2 * * *" # Daily at 2 AM
) -> "SemanticCompactionV4":
    """
    Registers the compaction process with HermesCronSchedulerV3.
    """
    compactor = SemanticCompactionV4(episodic_memory, semantic_memory, llm_client)

    @scheduler.task(cron_expr, name="semantic_compaction_v4")
    async def compact_task():
        await compactor.run_compaction()

    return compactor
