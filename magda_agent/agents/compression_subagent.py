import asyncio
import logging
from typing import Optional, List, Dict, Any
from magda_agent.llm_client import LLMClient
from magda_agent.memory.episodic import EpisodicMemory

class ContextCompressionSubagent:
    """
    A lightweight subagent that continuously runs in the background to compress
    and summarize episodic memories to optimize long-term context storage.
    Inspired by Claude Context Compression trends.
    """

    def __init__(
        self,
        llm: LLMClient,
        episodic_memory: EpisodicMemory,
        batch_size: int = 10,
        sleep_interval: float = 60.0
    ):
        """
        Initializes the ContextCompressionSubagent.

        Args:
            llm: Language Model client to be used for summarization.
            episodic_memory: The episodic memory instance to compress.
            batch_size: The maximum number of events to fetch per compression cycle.
            sleep_interval: Time to wait (in seconds) between compression cycles.
        """
        self.llm = llm
        self.episodic_memory = episodic_memory
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Starts the continuous compression loop in the background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self.run_compression_loop())
        logging.info("ContextCompressionSubagent started.")

    async def stop(self) -> None:
        """Stops the continuous compression loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logging.info("ContextCompressionSubagent stopped.")

    async def run_compression_loop(self) -> None:
        """
        The main loop that continuously fetches uncompressed memories,
        summarizes them, stores the summary, and marks original events as decayed.
        """
        while self._running:
            try:
                await self.compress_next_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in context compression loop: {e}")

            await asyncio.sleep(self.sleep_interval)

    async def compress_next_batch(self) -> None:
        """
        Fetches the next batch of undecayed events, groups them,
        summarizes them using the LLM, and stores the new summary.
        """
        events = self.episodic_memory.get_all_events(
            include_decayed=False,
            limit=self.batch_size
        )

        if not events or len(events) < 2:
            # Not enough events to compress
            return

        logging.info(f"Compressing batch of {len(events)} episodic memories.")

        # Group events by user_id if possible, or process them as a single batch
        user_groups: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for event in events:
            user_id = event.get("metadata", {}).get("user_id")
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(event)

        for user_id, user_events in user_groups.items():
            if len(user_events) < 2:
                continue

            events_text = "\n".join([f"- {e['text']}" for e in user_events])
            prompt = (
                "Please concisely summarize the following episodic memory events. "
                "Retain the most critical information, entities, and context:\n\n"
                f"{events_text}"
            )

            messages = [
                {"role": "system", "content": "You are a context compression subagent. Summarize the user's episodic memories accurately and concisely."},
                {"role": "user", "content": prompt}
            ]

            try:
                summary = await self.llm.chat_completion(messages, temperature=0.3)
                summary = summary.strip()

                if summary:
                    # Store the new summarized event
                    self.episodic_memory.store_event(
                        text=summary,
                        metadata={"type": "compressed_summary"},
                        user_id=user_id
                    )

                    # Mark original events as decayed
                    for event in user_events:
                        self.episodic_memory.decay_event(event["id"])

                    logging.info(f"Successfully compressed {len(user_events)} events for user_id={user_id}.")
            except Exception as e:
                logging.error(f"Failed to compress events for user_id={user_id}: {e}")
