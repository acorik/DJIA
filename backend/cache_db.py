"""Caché local (SQLite) de análisis de audio, indexada por hash del contenido real
del archivo — no por nombre ni ruta — para no repetir el cálculo DSP/GPU al
recargar la misma carpeta."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_analysis (
    file_hash   TEXT PRIMARY KEY,
    file_path   TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_cached(file_hash: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_json FROM track_analysis WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return json.loads(row[0]) if row else None


def put_cached(file_hash: str, file_path: str, payload: dict) -> None:
    import time
    with _connect() as conn:
        conn.execute(
            """INSERT INTO track_analysis (file_hash, file_path, payload_json, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(file_hash) DO UPDATE SET
                   file_path = excluded.file_path,
                   payload_json = excluded.payload_json,
                   created_at = excluded.created_at
            """,
            (file_hash, file_path, json.dumps(payload), time.time()),
        )


def clear_cache() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM track_analysis")
