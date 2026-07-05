import sqlite3
import json
from typing import Dict, Any, List, Optional
import os


class LongitudinalTracker:
    """
    A longitudinal tracker for agent-generated code.
    Records PR outcomes and test failures in an SQLite database.
    """
    def __init__(self, db_path: str = "longitudinal_metrics.db", max_entries: int = 1000):
        """
        Initializes the LongitudinalTracker.

        Args:
            db_path (str): The path to the SQLite database file.
            max_entries (int): The maximum number of entries to store.
        """
        self.db_path = db_path
        self.max_entries = max_entries
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pr_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pr_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    test_failures INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def record_pr_outcome(self, pr_id: str, outcome: str, test_failures: int) -> None:
        """
        Records the outcome of a PR and its test failures.

        Args:
            pr_id (str): The identifier of the Pull Request.
            outcome (str): The outcome of the PR (e.g., 'success', 'failure').
            test_failures (int): The number of test failures in the PR.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pr_metrics (pr_id, outcome, test_failures)
                VALUES (?, ?, ?)
            ''', (pr_id, outcome, test_failures))

            # Enforce bounds
            cursor.execute('''
                DELETE FROM pr_metrics
                WHERE id NOT IN (
                    SELECT id FROM pr_metrics
                    ORDER BY id DESC
                    LIMIT ?
                )
            ''', (self.max_entries,))

            conn.commit()

    def get_summary(self) -> Dict[str, Any]:
        """
        Retrieves a summary of the longitudinal metrics.

        Returns:
            Dict[str, Any]: A dictionary containing total PRs, success rate, and total test failures.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*), SUM(CASE WHEN outcome = "success" THEN 1 ELSE 0 END), SUM(test_failures) FROM pr_metrics')
            row = cursor.fetchone()

            total_prs = row[0] if row[0] is not None else 0
            successful_prs = row[1] if row[1] is not None else 0
            total_failures = row[2] if row[2] is not None else 0

            success_rate = (successful_prs / total_prs) * 100 if total_prs > 0 else 0.0

            return {
                "total_prs": total_prs,
                "success_rate": round(success_rate, 2),
                "total_test_failures": total_failures
            }
