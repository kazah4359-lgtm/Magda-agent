import logging
from typing import Optional, Coroutine, Any, Callable

from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

logger = logging.getLogger(__name__)

class DailyReportManager:
    """
    Manages the scheduled daily reports generation using HermesCronSchedulerV3.
    """

    def __init__(self, scheduler: Optional[HermesCronSchedulerV3] = None):
        """
        Initializes the DailyReportManager.

        Args:
            scheduler: The CronScheduler instance. If None, a new one is created using memory db.
        """
        self.scheduler = scheduler or HermesCronSchedulerV3()

    def register_daily_report(self, name: str, report_func: Callable[..., Coroutine[Any, Any, str]], cron_expr: str = "0 8 * * *") -> None:
        """
        Registers an async function to be executed as a daily report task.

        Args:
            name: The name of the report task.
            report_func: The async function that generates the report. Should return a string.
            cron_expr: The cron expression indicating when to generate the report. Defaults to "0 8 * * *" (8:00 AM daily).
        """
        self.scheduler.schedule(cron_expr, report_func, name=name)
        logger.info(f"Registered daily report '{name}' with schedule '{cron_expr}'")

    async def generate_report_now(self, name: str) -> Optional[str]:
        """
        Immediately runs a report task that has been registered.

        Args:
            name: The name of the report task to run immediately.

        Returns:
            The generated report string, or None if the report is not registered or an error occurred.
        """
        func = self.scheduler._func_registry.get(name)
        if not func:
            logger.error(f"Cannot generate report '{name}', it is not registered.")
            return None

        logger.info(f"Generating daily report '{name}' immediately.")
        try:
            result = await func()
            return result
        except Exception as e:
            logger.error(f"Error generating daily report '{name}': {e}", exc_info=True)
            return None

    async def start(self) -> None:
        """
        Starts the underlying scheduler loop.
        """
        logger.info("Starting DailyReportManager scheduler")
        await self.scheduler.start()

    async def stop(self) -> None:
        """
        Stops the underlying scheduler loop.
        """
        logger.info("Stopping DailyReportManager scheduler")
        await self.scheduler.stop()
