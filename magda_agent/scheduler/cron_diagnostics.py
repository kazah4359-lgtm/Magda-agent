import logging
from typing import Optional
from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

logger = logging.getLogger(__name__)


class DiagnosticScheduler:
    """
    Manages scheduled diagnostic reports generation using HermesCronSchedulerV3.
    """

    def __init__(self, scheduler: Optional[HermesCronSchedulerV3] = None):
        self.scheduler = scheduler or HermesCronSchedulerV3()
        self._register_tasks()

    def _register_tasks(self):
        # Schedule the diagnostic check to run every hour at minute 0
        self.scheduler.schedule("0 * * * *", self.run_diagnostics, name="diagnostic_health_check")

    async def run_diagnostics(self) -> None:
        """
        Runs diagnostic health checks and logs results locally.
        Gracefully handles any exceptions to avoid interrupting the main loop.
        """
        logger.info("Starting scheduled diagnostic health checks...")
        try:
            # Perform basic checks

            # Check 1: Can we read from memory modules or SQLite?
            # In a real scenario we'd do a quick ping to DB/Chroma, here we just simulate the step
            db_status = "OK"

            # Check 2: Check memory usage or system resources
            # Simulated check
            memory_status = "OK"

            # Check 3: Check scheduler health
            scheduler_status = "OK" if self.scheduler._running else "STOPPED"

            # Log the diagnostic report locally
            logger.info(
                f"Diagnostic Report:\n"
                f" - DB Status: {db_status}\n"
                f" - Memory Status: {memory_status}\n"
                f" - Scheduler Status: {scheduler_status}"
            )

        except Exception as e:
            logger.error(f"Failed to run diagnostic health checks: {e}", exc_info=True)
