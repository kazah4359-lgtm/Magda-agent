import sqlite3
import json
import time
import logging
from typing import Dict, Any, Optional

class ACSPersistence:
    """
    Persists ACS (Agent Control Specification) validation checkpoint states to SQLite.
    Allows for audit trails and runtime policy adjustments by recording the outcome
    of each of the 5 validation checkpoints.
    """

    def __init__(self, db_path: str = "acs_persistence.db") -> None:
        """
        Initializes the ACSPersistence layer.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS acs_checkpoints_log (
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
            self.logger.error(f"Failed to initialize ACS persistence database at {self.db_path}: {e}")

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
                    INSERT INTO acs_checkpoints_log (timestamp, checkpoint_id, status, reason, workflow_context)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (timestamp, checkpoint_id, status, reason, context_json)
                )
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to log ACS checkpoint {checkpoint_id} to database: {e}")

    def get_logs(self, checkpoint_id: Optional[int] = None) -> list:
        """
        Retrieves logs from the database.

        Args:
            checkpoint_id: Optional filter by checkpoint ID.

        Returns:
            A list of log entries.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if checkpoint_id:
                    cursor.execute("SELECT * FROM acs_checkpoints_log WHERE checkpoint_id = ?", (checkpoint_id,))
                else:
                    cursor.execute("SELECT * FROM acs_checkpoints_log")
                return cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to retrieve ACS logs: {e}")
            return []
