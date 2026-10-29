"""Disposable analysis cache, separate from every .shotlab project."""

import json
import os
from pathlib import Path
import sqlite3
import zlib


def default_cache_directory():
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return base / "ShotLab" / "ShotBrowser" / "cache" / "statistics"


class ColorCache:
    """One connection per analysis worker; only complete result groups are published."""

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else default_cache_directory()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.directory / "colors-v1.sqlite3", timeout=10)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS features (
                source TEXT, method TEXT, frame INTEGER, payload BLOB,
                PRIMARY KEY(source, method, frame));
            CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload BLOB);
        """)
        self.pending = 0

    @staticmethod
    def encode(value):
        return zlib.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"))

    @staticmethod
    def decode(payload):
        return json.loads(zlib.decompress(payload)) if payload is not None else None

    def result(self, key):
        row = self.connection.execute("SELECT payload FROM results WHERE key=?", (key,)).fetchone()
        return self.decode(row[0]) if row else None

    def publish(self, key, value):
        self.connection.execute("INSERT OR REPLACE INTO results VALUES (?,?)", (key, self.encode(value)))
        self.connection.commit()
        self.pending = 0

    def feature(self, source, method, frame):
        row = self.connection.execute("SELECT payload FROM features WHERE source=? AND method=? AND frame=?",
                                      (source, method, frame)).fetchone()
        return self.decode(row[0]) if row else None

    def remember(self, source, method, frame, feature):
        self.connection.execute("INSERT OR REPLACE INTO features VALUES (?,?,?,?)", (source, method, frame, self.encode(feature)))
        self.pending += 1
        if self.pending >= 32:
            self.connection.commit()
            self.pending = 0

    def close(self):
        self.connection.commit()  # Valid completed samples survive cancellation.
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
