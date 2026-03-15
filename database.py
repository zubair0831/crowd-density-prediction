# ═══════════════════════════════════════════════════════════════════════════════
# database.py — SQLite persistence for sessions
# ═══════════════════════════════════════════════════════════════════════════════

import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = "sessions.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
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
    """)
    conn.commit()
    conn.close()


def db_upsert_session(s: Dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO sessions
        (id,name,venue_name,created_at,total_frames,duration_sec,
         peak_count,avg_count,alert_count,peak_behavior,calibration,summary)
        VALUES (:id,:name,:venue_name,:created_at,:total_frames,:duration_sec,
                :peak_count,:avg_count,:alert_count,:peak_behavior,:calibration,:summary)
    """, s)
    conn.commit()
    conn.close()


def db_list_sessions() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id,name,venue_name,created_at,total_frames,peak_count,"
        "avg_count,alert_count,peak_behavior FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_get_session(sid: str) -> Optional[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row  = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def db_delete_session(sid: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0