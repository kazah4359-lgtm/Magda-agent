import sqlite3
import logging
from typing import Optional, List, Dict, Any

class LongitudinalMetrics:
    """
    LongitudinalMetrics logs continuous improvement metrics specifically for
    test success rates over time.
    It stores these metrics persistently in a SQLite database.
    """
    def __init__(self, db_path: str = "./metrics_db.sqlite3") -> None:
        """
        Initialize the LongitudinalMetrics with a SQLite database.

        Args:
            db_path (str): Path to store the DB, or ':memory:' for an ephemeral database.
        """
        self.db_path = db_path
        # If it's an in-memory DB, we must keep the connection open to retain data
        self._memory_conn = sqlite3.connect(':memory:') if self.db_path == ':memory:' else None
        self._init_db()
        logging.info(f"Initialized LongitudinalMetrics with database: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the database."""
        if self._memory_conn:
            return self._memory_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initializes the database schema for longitudinal tracking."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_success_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_suite TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            if not self._memory_conn:
                conn.close()

    def log_test_success_rate(self, test_suite: str, total_tests: int, passed_tests: int) -> None:
        """
        Logs the test success rate for a given test suite execution.

        Args:
            test_suite (str): The name or identifier of the test suite.
            total_tests (int): The total number of tests executed.
            passed_tests (int): The number of tests that passed.
        """
        if total_tests <= 0:
            logging.warning("Cannot log success rate with total_tests <= 0")
            return

        success_rate = (passed_tests / total_tests) * 100.0

        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO test_success_rates
                       (test_suite, success_rate, total_tests, passed_tests)
                       VALUES (?, ?, ?, ?)''',
                    (test_suite, success_rate, total_tests, passed_tests)
                )
                conn.commit()
            finally:
                if not self._memory_conn:
                    conn.close()
            logging.debug(f"Logged test success rate for '{test_suite}': {success_rate:.2f}%")
        except Exception as e:
            logging.error(f"Failed to log test success rate for '{test_suite}': {e}")

    def get_success_rates(self, test_suite: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves recent success rates for a specific test suite.

        Args:
            test_suite (str): The name of the test suite to retrieve.
            limit (int): The maximum number of entries to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing success rate data.
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT success_rate, total_tests, passed_tests, timestamp
                       FROM test_success_rates
                       WHERE test_suite = ?
                       ORDER BY timestamp DESC LIMIT ?''',
                    (test_suite, limit)
                )
                rows = cursor.fetchall()
            finally:
                if not self._memory_conn:
                    conn.close()

            results = []
            for row in rows:
                success_rate, total_tests, passed_tests, timestamp = row
                results.append({
                    'success_rate': success_rate,
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'timestamp': timestamp
                })
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve success rates for '{test_suite}': {e}")
            return []
