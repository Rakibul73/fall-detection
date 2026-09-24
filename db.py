import sqlite3
import os

DB_PATH = "data/events.db"


def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS falls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            timestamp TEXT,
            confidence REAL,
            clip_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_fall(track_id, timestamp, confidence, clip_path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO falls (track_id, timestamp, confidence, clip_path) VALUES (?, ?, ?, ?)",
        (track_id, timestamp, confidence, clip_path)
    )
    conn.commit()
    conn.close()
