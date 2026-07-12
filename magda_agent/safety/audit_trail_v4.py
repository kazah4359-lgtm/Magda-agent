import time
import json
import sqlite3
import functools
import inspect
import copy
from typing import Any, Callable, Dict, List, Optional


class AuditTrailV4:
    """
    Agent Guard Audit Trail v4.
    Inspired by Prempti (Falco) trend, this module provides an audit trail
    that intercepts all tool calls, securely logs parameters and results
    (scrubbing sensitive information), and stores them in an SQLite database
    for offline review. Supports both synchronous and asynchronous tools.
    """

    def __init__(self, db_path: str = "audit_trail_v4.db") -> None:
        """
        Initializes the AuditTrailV4 with an SQLite database.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite database schema if not already initialized."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS audit_logs_v4 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        tool_name TEXT NOT NULL,
                        args TEXT NOT NULL,
                        kwargs TEXT NOT NULL,
                        result TEXT NOT NULL,
                        duration REAL NOT NULL
                    )
                    '''
                )
                conn.commit()
        except sqlite3.Error as e:
            import logging
            logging.error(f"AuditTrailV4 DB Initialization failed at {self.db_path}: {e}")

    def _sanitize(self, data: Any) -> Any:
        """
        Recursively scrubs sensitive information (passwords, keys, tokens, etc.)
        from the data before logging.
        """
        if inspect.isawaitable(data):
            return "<awaitable>"

        # Prevent infinite recursion on self-referential structures
        memo = {}

        def _deep_sanitize(obj: Any) -> Any:
            obj_id = id(obj)
            if obj_id in memo:
                return "<recursive reference>"

            if isinstance(obj, dict):
                memo[obj_id] = True
                sanitized = {}
                for k, v in obj.items():
                    k_str = str(k).lower()
                    # Check for sensitive keywords in the key
                    is_sensitive = any(
                        s in k_str for s in ["password", "api_key", "token", "secret", "auth", "credential", "private", "key"]
                    )

                    if is_sensitive:
                        # Redact primitive types or full complex types completely to be safe
                        if isinstance(v, (dict, list)):
                            # Sanitize inside just in case or fully redact?
                            # Safe approach: fully redact. But let's redact values of list/dict
                            if isinstance(v, list):
                                sanitized[k] = ["***" for _ in v]
                            else:
                                sanitized[k] = "***"
                        else:
                            sanitized[k] = "***"
                    else:
                        sanitized[k] = _deep_sanitize(v)
                return sanitized

            elif isinstance(obj, list):
                memo[obj_id] = True
                return [_deep_sanitize(item) for item in obj]

            elif isinstance(obj, tuple):
                memo[obj_id] = True
                return tuple(_deep_sanitize(item) for item in obj)

            try:
                return copy.deepcopy(obj)
            except Exception:
                return f"<uncopiable: {type(obj).__name__}>"

        return _deep_sanitize(data)

    def _log_to_db(self, tool_name: str, args: tuple, kwargs: dict, result: Any, duration: float) -> None:
        """
        Serializes and saves the sanitized execution data to SQLite.
        """
        timestamp = time.time()

        # Sanitize data
        safe_args = self._sanitize(list(args))
        safe_kwargs = self._sanitize(kwargs)
        safe_result = self._sanitize(result)

        # JSON serialize safely
        def safe_dumps(obj):
            try:
                return json.dumps(obj)
            except (TypeError, ValueError):
                return str(obj)

        args_json = safe_dumps(safe_args)
        kwargs_json = safe_dumps(safe_kwargs)
        result_json = safe_dumps(safe_result)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO audit_logs_v4 (timestamp, tool_name, args, kwargs, result, duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (timestamp, tool_name, args_json, kwargs_json, result_json, duration)
                )
                conn.commit()
        except sqlite3.Error as e:
            import logging
            logging.error(f"AuditTrailV4 DB logging failed: {e}")

    def intercept(self, tool_name: Optional[str] = None) -> Callable:
        """
        A decorator to intercept tool calls.
        It measures execution time, captures input/output, sanitizes them,
        and stores the audit log securely offline.

        Args:
            tool_name: An optional explicit name for the tool. Defaults to the function's name.
        """
        def decorator(func: Callable) -> Callable:
            name_to_use = tool_name or func.__name__

            if inspect.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    result = "<uninitialized_or_cancelled>"
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except BaseException as e:
                        result = f"Error: {str(e)}"
                        raise
                    finally:
                        duration = time.time() - start_time
                        self._log_to_db(name_to_use, args, kwargs, result, duration)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.time()
                    result = "<uninitialized_or_cancelled>"
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except BaseException as e:
                        result = f"Error: {str(e)}"
                        raise
                    finally:
                        duration = time.time() - start_time
                        self._log_to_db(name_to_use, args, kwargs, result, duration)
                return sync_wrapper

        return decorator

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieves all audit logs from the database.

        Returns:
            A list of dictionaries, where each dict represents an audit log entry.
        """
        logs = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, timestamp, tool_name, args, kwargs, result, duration FROM audit_logs_v4 ORDER BY timestamp ASC"
                )
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        args_obj = json.loads(row[3])
                    except (ValueError, TypeError):
                        args_obj = row[3]

                    try:
                        kwargs_obj = json.loads(row[4])
                    except (ValueError, TypeError):
                        kwargs_obj = row[4]

                    try:
                        result_obj = json.loads(row[5])
                    except (ValueError, TypeError):
                        result_obj = row[5]

                    logs.append({
                        "id": row[0],
                        "timestamp": row[1],
                        "tool_name": row[2],
                        "args": args_obj,
                        "kwargs": kwargs_obj,
                        "result": result_obj,
                        "duration": row[6]
                    })
        except sqlite3.Error as e:
            import logging
            logging.error(f"AuditTrailV4 DB query failed: {e}")

        return logs
