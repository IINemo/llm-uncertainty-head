import sqlite3
import pickle
from pathlib import Path


class DiskCache:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)
            conn.commit()

    def _get_connection(self):
        return sqlite3.connect(self.cache_path, check_same_thread=False)

    def get(self, key, func: callable = lambda: None):
        key = str(key)
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row is not None:
                return pickle.loads(row[0])

            # If not cached, compute and store the value
            value = func()
            conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                         (key, pickle.dumps(value)))
            conn.commit()
            return value

    def contains(self, key):
        key = str(key)
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row is not None

    def __len__(self):
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            return cursor.fetchone()[0]

    def __iter__(self):
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT key, value FROM cache")
            for row in cursor:
                yield row[0], pickle.loads(row[1])
