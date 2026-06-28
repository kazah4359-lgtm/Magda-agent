import time
import copy
import json
import sqlite3
from collections import deque
from typing import Any, Deque, Dict, List, Optional


class AuditTrail:
    """
    Records and queries tool invocations: what, when, why (from plan/checkpoints),
    result, and duration. Includes recursive sanitization to prevent sensitive
    data logging and uses a bounded ring buffer to avoid memory leaks.
    Inspired by Prempti (Falco).
    """

    def __init__(self, max_capacity: int = 1000, db_path: Optional[str] = "audit_trail.db") -> None:
        """
        Initializes the AuditTrail with a fixed capacity and an SQLite database for persistence.

        Args:
            max_capacity: Maximum number of entries to keep in the in-memory trail.
            db_path: Path to the SQLite database file. If None, only in-memory logging is used.
        """
        self.max_capacity = max_capacity
        self.trail: Deque[Dict[str, Any]] = deque(maxlen=max_capacity)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite database schema if a path is provided."""
        if not self.db_path:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        tool_name TEXT NOT NULL,
                        kwargs TEXT NOT NULL,
                        why TEXT NOT NULL,
                        result TEXT NOT NULL,
                        duration REAL NOT NULL
                    )
                    '''
                )
                conn.commit()
        except sqlite3.Error as e:
            import logging
            logging.error(f"Failed to initialize SQLite audit database at {self.db_path}: {e}")

    def _sanitize(self, data: Any) -> Any:
        """
        Recursively sanitizes sensitive data from dictionaries and lists.
        Redacts values for keys that match sensitive patterns.
        """
        import inspect
        if inspect.isawaitable(data):
            return "<awaitable>"

        if isinstance(data, dict):
            sanitized = {}
            for k, v in data.items():
                k_l = k.lower()
                is_s = any(s in k_l for s in ["password", "api_key", "auth_header", "token", "access_key", "secret"])
                if not is_s:
                    is_s = k_l in {"key", "auth", "credential", "private"}

                if is_s:
                    if isinstance(v, dict):
                        # If a sensitive key points to a dict, we redact the whole thing
                        # to be safe, or we could continue to sanitize it.
                        # Redacting the whole thing is safer.
                        sanitized[k] = "***"
                    elif isinstance(v, list):
                        # Redact all primitives in the list, recurse for dicts
                        sanitized[k] = ["***" if not isinstance(item, (dict, list)) else self._sanitize(item) for item in v]
                    else:
                        sanitized[k] = "***"
                else:
                    sanitized[k] = self._sanitize(v)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return copy.deepcopy(data)

    def log_call(self, tool_name: str, kwargs: Dict[str, Any], why: str, result: Any, duration: float = 0.0) -> None:
        """
        Logs a tool call with metadata and sanitization, both in memory and to SQLite.

        Args:
            tool_name: Name of the tool or action.
            kwargs: Arguments passed to the tool.
            why: Reason for the call or outcome.
            result: Outcome of the execution or "allowed"/"blocked".
            duration: Time taken in seconds.
        """
        timestamp = time.time()
        sanitized_kwargs = self._sanitize(kwargs)
        sanitized_result = self._sanitize(result)

        entry = {
            "timestamp": timestamp,
            "tool_name": tool_name,
            "kwargs": sanitized_kwargs,
            "why": why,
            "result": sanitized_result,
            "duration": duration
        }
        self.trail.append(entry)

        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        '''
                        INSERT INTO audit_logs (timestamp, tool_name, kwargs, why, result, duration)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            timestamp,
                            tool_name,
                            json.dumps(sanitized_kwargs),
                            why,
                            json.dumps(sanitized_result) if not isinstance(sanitized_result, str) else sanitized_result,
                            duration
                        )
                    )
                    conn.commit()
            except sqlite3.Error as e:
                import logging
                logging.error(f"Failed to log to SQLite audit database at {self.db_path}: {e}")

    def query(self, tool_name: Optional[str] = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Queries the audit trail with optional filters from the in-memory trail.
        """
        results = list(self.trail)
        if tool_name:
            results = [e for e in results if e["tool_name"] == tool_name]
        if start_time is not None:
            results = [e for e in results if e["timestamp"] >= start_time]
        if end_time is not None:
            results = [e for e in results if e["timestamp"] <= end_time]
        return results

    def query_db(self, tool_name: Optional[str] = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Queries the audit trail from the SQLite database.
        """
        if not self.db_path:
            return []

        results = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT timestamp, tool_name, kwargs, why, result, duration FROM audit_logs WHERE 1=1"
                params = []

                if tool_name:
                    query += " AND tool_name = ?"
                    params.append(tool_name)
                if start_time is not None:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                if end_time is not None:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()

                for row in rows:
                    try:
                        kwargs_dict = json.loads(row[2])
                    except json.JSONDecodeError:
                        kwargs_dict = row[2]

                    try:
                        result_obj = json.loads(row[4])
                    except (json.JSONDecodeError, TypeError):
                        result_obj = row[4]

                    results.append({
                        "timestamp": row[0],
                        "tool_name": row[1],
                        "kwargs": kwargs_dict,
                        "why": row[3],
                        "result": result_obj,
                        "duration": row[5]
                    })
        except sqlite3.Error as e:
            import logging
            logging.error(f"Failed to query SQLite audit database at {self.db_path}: {e}")

        return results

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all entries in the audit trail."""
        return list(self.trail)

    def clear(self) -> None:
        """Clears all entries from the in-memory audit trail and the SQLite database."""
        self.trail.clear()

        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM audit_logs")
                    conn.commit()
            except sqlite3.Error as e:
                import logging
                logging.error(f"Failed to clear SQLite audit database at {self.db_path}: {e}")
