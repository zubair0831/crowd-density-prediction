# ═══════════════════════════════════════════════════════════════════════════════
# database.py — SQLite persistence  v3.2
# ═══════════════════════════════════════════════════════════════════════════════
# Changes vs v3.1:
#   • _db() context-manager — every caller gets auto-commit/close
#   • WAL journal mode for concurrent-reader safety
#   • Row-factory set once at connection time
# ═══════════════════════════════════════════════════════════════════════════════

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

DB_PATH = "sessions.db"

_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS sessions (
        id            TEXT PRIMARY KEY,
        name          TEXT,
        venue_name    TEXT,
        created_at    TEXT,
        total_frames  INTEGER DEFAULT 0,
        duration_sec  REAL    DEFAULT 0,
        peak_count    INTEGER DEFAULT 0,
        avg_count     REAL    DEFAULT 0,
        alert_count   INTEGER DEFAULT 0,
        peak_behavior TEXT    DEFAULT 'NORMAL',
        calibration   TEXT,
        summary       TEXT
    )
"""

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _db() as conn:
        conn.execute(_CREATE_SQL)


def db_upsert_session(s: Dict[str, Any]) -> None:
    with _db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (id,name,venue_name,created_at,total_frames,duration_sec,
             peak_count,avg_count,alert_count,peak_behavior,calibration,summary)
            VALUES (:id,:name,:venue_name,:created_at,:total_frames,:duration_sec,
                    :peak_count,:avg_count,:alert_count,:peak_behavior,:calibration,:summary)
        """, s)


def db_list_sessions() -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT id,name,venue_name,created_at,total_frames,peak_count,"
            "avg_count,alert_count,peak_behavior FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def db_get_session(sid: str) -> Optional[Dict]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None


def db_delete_session(sid: str) -> bool:
    with _db() as conn:
        c = conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    return c.rowcount > 0