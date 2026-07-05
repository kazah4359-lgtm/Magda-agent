import logging
import sqlite3
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Coroutine, Any, Callable

from magda_agent.operations.cron_v3 import HermesCronSchedulerV3

logger = logging.getLogger(__name__)

class SQLiteCronBackupManager:
    """
    Manages the scheduled backup operations for SQLite databases using HermesCronSchedulerV3.
    """

    def __init__(
        self,
        databases: List[str],
        backup_dir: str,
        scheduler: Optional[HermesCronSchedulerV3] = None
    ):
        """
        Initializes the SQLiteCronBackupManager.

        Args:
            databases: A list of paths to the SQLite databases to backup.
            backup_dir: The directory where backups will be stored.
            scheduler: The CronScheduler instance. If None, a new one is created using memory db.
        """
        self.databases = databases
        self.backup_dir = backup_dir
        self.scheduler = scheduler or HermesCronSchedulerV3()

    async def backup_databases(self) -> None:
        """
        Performs the backup operation for all registered databases.
        Copies the databases to the backup directory using sqlite3 backup API.
        """
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir, exist_ok=True)

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for db_path in self.databases:
            if not os.path.exists(db_path):
                logger.warning(f"Database {db_path} does not exist, skipping backup.")
                continue

            db_name = os.path.basename(db_path)
            backup_path = os.path.join(self.backup_dir, f"{db_name}.{now_str}.bak")

            try:
                # Use sqlite3.Connection.backup for safe online backups
                def _do_backup() -> None:
                    src_conn = sqlite3.connect(db_path)
                    dst_conn = sqlite3.connect(backup_path)
                    with src_conn, dst_conn:
                        src_conn.backup(dst_conn)
                    src_conn.close()
                    dst_conn.close()

                await asyncio.to_thread(_do_backup)

                logger.info(f"Successfully backed up {db_path} to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to backup {db_path} to {backup_path}: {e}")

    def register_nightly_backup(self, cron_expr: str = "0 2 * * *") -> None:
        """
        Registers the backup operation as a nightly task.

        Args:
            cron_expr: The cron expression indicating when to run the backup. Defaults to "0 2 * * *" (2:00 AM daily).
        """
        self.scheduler.schedule(cron_expr, self.backup_databases, name="sqlite_nightly_backup")
        logger.info(f"Registered nightly backup with schedule '{cron_expr}'")

    async def start(self) -> None:
        """
        Starts the underlying scheduler loop.
        """
        logger.info("Starting SQLiteCronBackupManager scheduler")
        await self.scheduler.start()

    async def stop(self) -> None:
        """
        Stops the underlying scheduler loop.
        """
        logger.info("Stopping SQLiteCronBackupManager scheduler")
        await self.scheduler.stop()
