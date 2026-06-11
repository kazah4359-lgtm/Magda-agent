import logging
from typing import Callable, Any, Coroutine, Dict
from magda_agent.operations.cron import OperationsCronScheduler

logger = logging.getLogger(__name__)

class DailyReportScheduler:
    """
    A scheduler dedicated to running daily reports and backups using the OperationsCronScheduler.
    """

    def __init__(self, scheduler: OperationsCronScheduler | None = None) -> None:
        """
        Initializes the DailyReportScheduler.

        Args:
            scheduler: The OperationsCronScheduler instance. If None, a new one is created.
        """
        self.scheduler = scheduler or OperationsCronScheduler()
        self._registered_reports: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

    def register_daily_report(self, name: str, report_func: Callable[..., Coroutine[Any, Any, Any]], cron_expr: str = "0 9 * * *") -> None:
        """
        Registers an async function to be executed as a daily report.

        Args:
            name: The name of the report.
            report_func: The async function that generates the report.
            cron_expr: The cron expression indicating when to run the report. Defaults to "0 9 * * *" (9:00 AM daily).
        """
        self._registered_reports[name] = report_func
        self.scheduler.schedule(cron_expr, report_func, name=name)
        logger.info(f"Registered daily report '{name}' with schedule '{cron_expr}'")

    def register_nightly_backup(self, name: str, backup_func: Callable[..., Coroutine[Any, Any, Any]], cron_expr: str = "0 2 * * *") -> None:
        """
        Registers an async function to be executed as a nightly backup.

        Args:
            name: The name of the backup task.
            backup_func: The async function that performs the backup.
            cron_expr: The cron expression indicating when to run the backup. Defaults to "0 2 * * *" (2:00 AM daily).
        """
        self._registered_reports[name] = backup_func
        self.scheduler.schedule(cron_expr, backup_func, name=name)
        logger.info(f"Registered nightly backup '{name}' with schedule '{cron_expr}'")

    async def start(self) -> None:
        """
        Starts the underlying scheduler.
        """
        logger.info("Starting DailyReportScheduler")
        await self.scheduler.start()

    async def stop(self) -> None:
        """
        Stops the underlying scheduler.
        """
        logger.info("Stopping DailyReportScheduler")
        await self.scheduler.stop()
