import sqlite3
import json
import logging
from typing import Optional, Dict, Any, List

class QualityTracker:
    """
    QualityTracker logs continuous improvement metrics like test pass rates and PR sizes.
    It stores these metrics persistently in a SQLite database so that Subconsciousness can read them for reflection.
    """
    def __init__(self, db_path: str = "./metrics_db.sqlite3") -> None:
        """
        Initialize the QualityTracker with a SQLite database.

        Args:
            db_path (str): Path to store the DB, or ':memory:' for an ephemeral database.
        """
        self.db_path = db_path
        # If it's an in-memory DB, we must keep the connection open to retain data
        self._memory_conn = sqlite3.connect(':memory:') if self.db_path == ':memory:' else None
        self._init_db()
        logging.info(f"Initialized QualityTracker with database: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the database."""
        if self._memory_conn:
            return self._memory_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initializes the database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            if not self._memory_conn:
                conn.close()

    def log_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs a specific metric value.

        Args:
            metric_name (str): The name of the metric (e.g., 'test_pass_rate', 'pr_size').
            value (float): The recorded value for the metric.
            metadata (Optional[Dict[str, Any]]): Additional context/metadata for the metric.
        """
        try:
            meta_json = json.dumps(metadata) if metadata else None
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO metrics (metric_name, value, metadata) VALUES (?, ?, ?)',
                    (metric_name, value, meta_json)
                )
                conn.commit()
            finally:
                if not self._memory_conn:
                    conn.close()
            logging.debug(f"Logged metric '{metric_name}': {value}")
        except Exception as e:
            logging.error(f"Failed to log metric '{metric_name}': {e}")

    def get_metrics(self, metric_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recent entries for a specific metric.

        Args:
            metric_name (str): The name of the metric to retrieve.
            limit (int): The maximum number of entries to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing metric metadata.
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT value, metadata, timestamp FROM metrics WHERE metric_name = ? ORDER BY timestamp DESC LIMIT ?',
                    (metric_name, limit)
                )
                rows = cursor.fetchall()
            finally:
                if not self._memory_conn:
                    conn.close()

            results = []
            for row in rows:
                value, meta_json, timestamp = row
                meta = json.loads(meta_json) if meta_json else {}
                meta['value'] = value
                meta['timestamp'] = timestamp
                results.append(meta)
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve metrics for '{metric_name}': {e}")
            return []

    def calculate_average(self, metric_name: str, limit: int = 10) -> Optional[float]:
        """
        Calculates the average value for a specific metric over recent entries.

        Args:
            metric_name (str): The name of the metric.
            limit (int): The number of recent entries to consider.

        Returns:
            Optional[float]: The average value, or None if no entries exist.
        """
        metrics = self.get_metrics(metric_name, limit=limit)
        if not metrics:
            return None

        total = sum(float(m.get("value", 0.0)) for m in metrics)
        return total / len(metrics)
