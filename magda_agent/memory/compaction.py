import asyncio
import logging
from typing import Optional
from magda_agent.memory.storage import MemorySystem

class MemoryCompactor:
    """
    Background routine that periodically compacts memory layers to save context space.
    It consolidates working memory and decays older episodic memory events.
    """

    def __init__(self, memory_system: MemorySystem, interval_seconds: int = 3600, episodic_limit: int = 1000):
        """
        Initialize the MemoryCompactor.

        Args:
            memory_system: The memory system instance to manage.
            interval_seconds: How often to run the compaction routine (in seconds). Default is 1 hour.
            episodic_limit: The maximum number of non-decayed events to keep per user before decaying the oldest.
        """
        self.memory_system = memory_system
        self.interval_seconds = interval_seconds
        self.episodic_limit = episodic_limit
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Starts the background compaction routine."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self.run_background_routine())
        logging.info("MemoryCompactor background routine started.")

    async def stop(self) -> None:
        """Stops the background compaction routine."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logging.info("MemoryCompactor background routine stopped.")

    async def run_background_routine(self) -> None:
        """
        The main background loop that performs the compaction operations.
        It triggers working memory consolidation and episodic memory decay.
        """
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)
                self.compact_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error during memory compaction routine: {e}")

    def compact_memory(self) -> None:
        """
        Executes a single pass of the memory compaction logic.
        """
        logging.info("Starting memory compaction pass.")

        # Consolidate working memory for all users currently active
        all_users = list(self.memory_system.working_memory._entries_by_user.keys())
        for u_id in all_users:
            try:
                self.memory_system.consolidate(user_id=u_id)
            except Exception as e:
                logging.error(f"Failed to consolidate memory for user {u_id}: {e}")

            # Also check and decay episodic memory for this user
            try:
                self._decay_episodic_for_user(u_id)
            except Exception as e:
                logging.error(f"Failed to decay episodic memory for user {u_id}: {e}")

        # Try to clean up any user-agnostic memories (user_id = None is treated as -1 in storage for working memory)
        try:
            self.memory_system.consolidate(user_id=None)
            self._decay_episodic_for_user(None)
        except Exception as e:
            logging.error(f"Failed to process generic memory compaction: {e}")

        logging.info("Memory compaction pass completed.")

    def _decay_episodic_for_user(self, user_id: Optional[int]) -> None:
        """
        Checks episodic memory limits for a user and decays oldest events if limit is exceeded.
        """
        if not hasattr(self.memory_system, 'episodic_memory'):
            return

        events = self.memory_system.episodic_memory.get_all_events(user_id=user_id, include_decayed=False, limit=self.episodic_limit + 100)

        # If the number of active events exceeds the limit, we need to decay the oldest ones.
        if len(events) > self.episodic_limit:
            # Assuming get_all_events returns them in insertion order (ChromaDB typically returns in an arbitrary or id order unless sorted,
            # but since we lack explicit timestamps, we decay based on the first returned items which represent the "excess")
            excess_count = len(events) - self.episodic_limit
            events_to_decay = events[:excess_count]

            for event in events_to_decay:
                event_id = event.get('id')
                if event_id:
                    self.memory_system.episodic_memory.decay_event(event_id)
                    logging.debug(f"MemoryCompactor decayed event {event_id} for user {user_id}")
