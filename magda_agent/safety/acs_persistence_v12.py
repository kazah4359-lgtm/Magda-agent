import sqlite3
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

class ACSPersistenceV12:
    """
    ACS (Agent Control Specification) Safety Checkpoints Persistence V12.
    Implements persistent SQL-based logging to record historical outcomes
    for the 5 standard safety checkpoints:
    1: Input Validation
    2: Intent Authorization
    3: Tool Policy
    4: State Transition
    5: Output Sanitization
    """

    def __init__(self, db_path: str = "acs_persistence_v12.db") -> None:
        """
        Initializes the ACSPersistenceV12 layer and creates the table if it doesn't exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite database schema for V12 log table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS acs_checkpoints_log_v12 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        checkpoint_id INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        reason TEXT,
                        workflow_context TEXT
                    )
                    '''
                )
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize ACS V12 persistence database at {self.db_path}: {e}")

    def log_checkpoint(
        self,
        checkpoint_id: int,
        status: str,
        reason: Optional[str] = None,
        workflow_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Logs the result of an ACS checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint (1-5).
            status: The status of the checkpoint ("passed" or "failed").
            reason: Optional reason for the status.
            workflow_context: Optional context data related to the workflow.
        """
        timestamp = time.time()
        context_json = json.dumps(workflow_context) if workflow_context else None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO acs_checkpoints_log_v12 (timestamp, checkpoint_id, status, reason, workflow_context)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (timestamp, checkpoint_id, status, reason, context_json)
                )
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to log ACS V12 checkpoint {checkpoint_id} to database: {e}")

    def get_logs(self, checkpoint_id: Optional[int] = None) -> List[Tuple[Any, ...]]:
        """
        Retrieves logs from the V12 database.

        Args:
            checkpoint_id: Optional filter by checkpoint ID (1-5).

        Returns:
            A list of log tuple entries.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if checkpoint_id is not None:
                    cursor.execute(
                        "SELECT id, timestamp, checkpoint_id, status, reason, workflow_context FROM acs_checkpoints_log_v12 WHERE checkpoint_id = ? ORDER BY id ASC",
                        (checkpoint_id,)
                    )
                else:
                    cursor.execute(
                        "SELECT id, timestamp, checkpoint_id, status, reason, workflow_context FROM acs_checkpoints_log_v12 ORDER BY id ASC"
                    )
                return cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve ACS V12 logs: {e}")
            return []

    def calculate_failure_rate(self, checkpoint_id: Optional[int] = None) -> float:
        """
        Calculates the failure rate of a specific checkpoint or all checkpoints.

        Args:
            checkpoint_id: Optional filter by checkpoint ID (1-5).

        Returns:
            Float representing the failure rate (from 0.0 to 1.0). Returns 0.0 if there are no logs.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if checkpoint_id is not None:
                    cursor.execute(
                        "SELECT COUNT(*) FROM acs_checkpoints_log_v12 WHERE checkpoint_id = ?",
                        (checkpoint_id,)
                    )
                    total = cursor.fetchone()[0]
                    if total == 0:
                        return 0.0
                    cursor.execute(
                        "SELECT COUNT(*) FROM acs_checkpoints_log_v12 WHERE checkpoint_id = ? AND status = 'failed'",
                        (checkpoint_id,)
                    )
                    failed = cursor.fetchone()[0]
                    return float(failed) / total
                else:
                    cursor.execute("SELECT COUNT(*) FROM acs_checkpoints_log_v12")
                    total = cursor.fetchone()[0]
                    if total == 0:
                        return 0.0
                    cursor.execute("SELECT COUNT(*) FROM acs_checkpoints_log_v12 WHERE status = 'failed'")
                    failed = cursor.fetchone()[0]
                    return float(failed) / total
        except sqlite3.Error as e:
            self.logger.error(f"Failed to calculate failure rate: {e}")
            return 0.0

    def count_total_runs(self) -> int:
        """
        Counts total checkpoint runs stored in the database.

        Returns:
            Total count of run entries.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM acs_checkpoints_log_v12")
                return int(cursor.fetchone()[0])
        except sqlite3.Error as e:
            self.logger.error(f"Failed to count total runs: {e}")
            return 0
