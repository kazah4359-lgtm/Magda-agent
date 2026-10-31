import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryCompactionCronV2:
    """
    OpenClaw Memory Compaction Cron V2.
    Inspired by OpenClaw virtual context: Schedules and executes an overnight
    cron job that compacts uncompressed episodic memory events into concise
    summaries, decays older raw events, and stores the consolidated knowledge.
    """

    def __init__(
        self,
        episodic_memory: Any,
        scheduler: Any,
        llm_client: Optional[Any] = None,
        cron_expr: str = "0 3 * * *",
        threshold: int = 10,
        batch_size: int = 10,
        task_name: str = "openclaw_memory_compaction_v2",
    ) -> None:
        """
        Initializes the MemoryCompactionCronV2.

        Args:
            episodic_memory: EpisodicMemory instance to compact.
            scheduler: Schedulers such as CronSchedulerV2, CronScheduler,
                       CronTaskManager, HermesCronSchedulerV3, etc.
            llm_client: Optional LLMClient or callable for summarization.
            cron_expr: Cron expression for scheduling (defaults to 3:00 AM daily: '0 3 * * *').
            threshold: Minimum number of active episodic events to trigger compaction.
            batch_size: Maximum number of old events to compact in a single pass.
            task_name: Unique task identifier in the scheduler.
        """
        self.episodic_memory = episodic_memory
        self.scheduler = scheduler
        self.llm_client = llm_client
        self.cron_expr = cron_expr
        self.threshold = threshold
        self.batch_size = batch_size
        self.task_name = task_name

    def schedule_job(self) -> None:
        """
        Registers the compaction task with the configured scheduler using duck typing.
        Supports CronSchedulerV2 (add_task), CronTaskManager/CronScheduler (register_task, schedule).
        """
        if hasattr(self.scheduler, "add_task"):
            self.scheduler.add_task(self.task_name, self.cron_expr, self.run_compaction)
        elif hasattr(self.scheduler, "register_task"):
            self.scheduler.register_task(self.task_name, self.cron_expr, self.run_compaction)
        elif hasattr(self.scheduler, "schedule"):
            self.scheduler.schedule(self.cron_expr, self.run_compaction, name=self.task_name)
        else:
            raise AttributeError("Provided scheduler does not support add_task, register_task, or schedule.")

        logger.info(f"Scheduled memory compaction job '{self.task_name}' with cron '{self.cron_expr}'.")

    async def _generate_summary(self, event_texts: List[str]) -> str:
        """
        Generates a summary of event texts using the provided LLM client or fallback heuristics.
        """
        combined = "\n".join(f"- {text}" for text in event_texts)
        if self.llm_client:
            prompt = (
                "You are an OpenClaw virtual context memory compaction engine. "
                "Synthesize the following episodic memory events into a coherent, concise summary "
                "preserving essential facts, learnings, and context:\n\n"
                f"{combined}"
            )
            try:
                if hasattr(self.llm_client, "generate"):
                    resp = await self.llm_client.generate(prompt)
                    if resp and not resp.startswith("Error:"):
                        return resp.strip()
                elif hasattr(self.llm_client, "chat_completion"):
                    resp = await self.llm_client.chat_completion(
                        [
                            {"role": "system", "content": "You are a memory compaction engine."},
                            {"role": "user", "content": prompt},
                        ]
                    )
                    if resp and not str(resp).startswith("Error:"):
                        return str(resp).strip()
                elif callable(self.llm_client):
                    res = self.llm_client(prompt)
                    if asyncio.iscoroutine(res):
                        res = await res
                    if res:
                        return str(res).strip()
            except Exception as e:
                logger.warning(f"LLM compaction failed, using fallback heuristic: {e}")

        # Fallback heuristic summary
        return f"Compacted summary of {len(event_texts)} events recorded up to {datetime.now().isoformat()}: " + "; ".join(
            t[:60] for t in event_texts[:3]
        )

    async def run_compaction(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes an overnight memory compaction pass:
        1. Reads un-decayed episodic memory events.
        2. Evaluates threshold. If below threshold, skips.
        3. Compacts oldest batch of events into a summary.
        4. Marks the compacted raw events as decayed.
        5. Stores the summary event in episodic memory.

        Returns:
            Dict containing the execution status and metrics.
        """
        logger.info(f"Starting OpenClaw memory compaction run for user {user_id}...")
        try:
            events = self.episodic_memory.get_all_events(
                user_id=user_id,
                include_decayed=False,
                limit=self.batch_size * 2,
            )

            if len(events) < self.threshold:
                logger.info(
                    f"Episodic memory count ({len(events)}) is below threshold ({self.threshold}). "
                    "Skipping compaction."
                )
                return {
                    "status": "skipped",
                    "reason": "below_threshold",
                    "events_count": len(events),
                    "compacted_count": 0,
                }

            events_to_compact = events[: self.batch_size]
            event_texts = [e["text"] for e in events_to_compact if "text" in e]

            summary = await self._generate_summary(event_texts)

            # Decay processed events
            decayed_count = 0
            for event in events_to_compact:
                event_id = event.get("id")
                if event_id:
                    self.episodic_memory.decay_event(event_id)
                    decayed_count += 1

            # Store the compacted summary
            self.episodic_memory.store_event(
                text=summary,
                metadata={
                    "type": "summary",
                    "source": "openclaw_memory_compaction_cron_v2",
                    "compacted_count": decayed_count,
                    "timestamp": datetime.now().isoformat(),
                },
                user_id=user_id,
            )

            logger.info(
                f"Successfully compacted {decayed_count} episodic memory events into summary: {summary[:50]}..."
            )
            return {
                "status": "success",
                "compacted_count": decayed_count,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Error executing memory compaction job: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "compacted_count": 0,
            }


# Backwards compatibility and alias
OpenClawMemoryCompactionCronV2 = MemoryCompactionCronV2


def schedule_openclaw_memory_compaction_v2(
    episodic_memory: Any,
    scheduler: Any,
    llm_client: Optional[Any] = None,
    cron_expr: str = "0 3 * * *",
    threshold: int = 10,
    batch_size: int = 10,
) -> MemoryCompactionCronV2:
    """
    Convenience function to instantiate and schedule OpenClaw memory compaction v2.
    """
    cron_job = MemoryCompactionCronV2(
        episodic_memory=episodic_memory,
        scheduler=scheduler,
        llm_client=llm_client,
        cron_expr=cron_expr,
        threshold=threshold,
        batch_size=batch_size,
    )
    cron_job.schedule_job()
    return cron_job
