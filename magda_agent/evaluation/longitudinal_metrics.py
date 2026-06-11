import sqlite3
import datetime
from typing import Dict, Any, List

class LongitudinalMetricsTracker:
    """
    Tracks and persists code quality metrics over time across multiple test runs.
    """
    def __init__(self, db_path: str = "./longitudinal_metrics_db.sqlite3"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initializes the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Records a single metric value.

        Args:
            metric_name (str): The name of the metric (e.g., 'test_coverage', 'complexity').
            value (float): The value of the metric.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.datetime.now(datetime.UTC).isoformat()
            cursor.execute(
                "INSERT INTO code_metrics (metric_name, value, timestamp) VALUES (?, ?, ?)",
                (metric_name, value, timestamp)
            )
            conn.commit()

    def get_metrics_history(self, metric_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the history of a specific metric.

        Args:
            metric_name (str): The name of the metric.
            limit (int): The maximum number of historical records to fetch.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing metric history.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value, timestamp FROM code_metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT ?",
                (metric_name, limit)
            )
            rows = cursor.fetchall()

        return [{"value": row[0], "timestamp": row[1]} for row in rows]
