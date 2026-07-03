import logging

from typing import Optional, Coroutine, Any, Callable

from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

logger = logging.getLogger(__name__)

class NightlyBackupManagerV2:
    """
    Manages the scheduled backup operations for memory layers using HermesCronSchedulerV3.
    """

    def __init__(self, scheduler: Optional[HermesCronSchedulerV3] = None):
        """
        Initializes the NightlyBackupManagerV2.

        Args:
            scheduler: The CronScheduler instance. If None, a new one is created using memory db.
        """
        self.scheduler = scheduler or HermesCronSchedulerV3()

    def register_nightly_backup(self, name: str, backup_func: Callable[..., Coroutine[Any, Any, Any]], cron_expr: str = "0 2 * * *") -> None:
        """
        Registers an async function to be executed as a nightly backup task.

        Args:
            name: The name of the backup task.
            backup_func: The async function that performs the backup.
            cron_expr: The cron expression indicating when to run the backup. Defaults to "0 2 * * *" (2:00 AM daily).
        """
        self.scheduler.schedule(cron_expr, backup_func, name=name)
        logger.info(f"Registered nightly backup '{name}' with schedule '{cron_expr}'")

    async def run_backup_now(self, name: str) -> None:
        """
        Immediately runs a backup task that has been registered.

        Args:
            name: The name of the backup task to run immediately.
        """
        func = self.scheduler._func_registry.get(name)
        if not func:
            logger.error(f"Cannot run backup '{name}', it is not registered.")
            return

        logger.info(f"Running backup task '{name}' immediately.")
        try:
            await func()
        except Exception as e:
            logger.error(f"Error running backup task '{name}': {e}", exc_info=True)

    async def start(self) -> None:
        """
        Starts the underlying scheduler loop.
        """
        logger.info("Starting NightlyBackupManagerV2 scheduler")
        await self.scheduler.start()

    async def stop(self) -> None:
        """
        Stops the underlying scheduler loop.
        """
        logger.info("Stopping NightlyBackupManagerV2 scheduler")
        await self.scheduler.stop()
