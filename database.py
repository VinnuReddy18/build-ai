"""
database.py — PostgreSQL / SQLite event logging for Home Guard AI.
"""

import os
import sqlite3
import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _use_postgres():
    """Return True if a PostgreSQL DATABASE_URL is configured."""
    return DATABASE_URL.startswith("postgresql")


def _get_pg_connection():
    """Return a psycopg2 connection."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def _get_sqlite_connection():
    """Return a sqlite3 connection (local fallback)."""
    conn = sqlite3.connect("security_events.db")
    conn.row_factory = sqlite3.Row
    return conn


def get_connection():
    """Return a database connection (Postgres or SQLite)."""
    if _use_postgres():
        return _get_pg_connection()
    return _get_sqlite_connection()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_PG_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS security_events (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMP DEFAULT NOW(),
    threat_level TEXT NOT NULL,
    event_description TEXT NOT NULL,
    image_data  TEXT
);
"""

_SQLITE_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS security_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT DEFAULT (datetime('now', 'localtime')),
    threat_level TEXT NOT NULL,
    event_description TEXT NOT NULL,
    image_data  TEXT
);
"""


def init_db():
    """Create the security_events table if it doesn't exist."""
    conn = get_connection()
    cur = conn.cursor()
    if _use_postgres():
        cur.execute(_PG_CREATE_TABLE)
    else:
        cur.execute(_SQLITE_CREATE_TABLE)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] security_events table ready.")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def log_event(threat_level: str, description: str, image_base64: str = None):
    """Insert a new security event into the database."""
    conn = get_connection()
    cur = conn.cursor()
    if _use_postgres():
        cur.execute(
            "INSERT INTO security_events (threat_level, event_description, image_data) "
            "VALUES (%s, %s, %s)",
            (threat_level, description, image_base64),
        )
    else:
        cur.execute(
            "INSERT INTO security_events (threat_level, event_description, image_data) "
            "VALUES (?, ?, ?)",
            (threat_level, description, image_base64),
        )
    conn.commit()
    cur.close()
    conn.close()


def get_recent_events(limit: int = 20):
    """Return the most recent events as a list of dicts."""
    conn = get_connection()
    cur = conn.cursor()
    if _use_postgres():
        cur.execute(
            "SELECT id, timestamp, threat_level, event_description, image_data "
            "FROM security_events ORDER BY timestamp DESC LIMIT %s",
            (limit,),
        )
    else:
        cur.execute(
            "SELECT id, timestamp, threat_level, event_description, image_data "
            "FROM security_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    events = []
    for row in rows:
        if _use_postgres():
            events.append({
                "id": row[0],
                "timestamp": row[1].strftime("%Y-%m-%d %H:%M:%S") if row[1] else "",
                "threat_level": row[2],
                "description": row[3],
                "image_data": row[4],
            })
        else:
            events.append({
                "id": row[0] if isinstance(row, tuple) else row["id"],
                "timestamp": row[1] if isinstance(row, tuple) else row["timestamp"],
                "threat_level": row[2] if isinstance(row, tuple) else row["threat_level"],
                "description": row[3] if isinstance(row, tuple) else row["event_description"],
                "image_data": row[4] if isinstance(row, tuple) else row["image_data"],
            })
    return events


def get_event_stats():
    """Return event counts grouped by threat_level."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT threat_level, COUNT(*) as cnt "
        "FROM security_events GROUP BY threat_level"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    stats = {"low": 0, "medium": 0, "high": 0, "total": 0}
    for row in rows:
        level = row[0] if isinstance(row, tuple) else row["threat_level"]
        count = row[1] if isinstance(row, tuple) else row["cnt"]
        level_lower = level.lower()
        if level_lower in stats:
            stats[level_lower] = count
        stats["total"] += count
    return stats


# Auto-init on import
init_db()
