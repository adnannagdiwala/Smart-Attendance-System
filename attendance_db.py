"""
Attendance Database Module
Handles logging and querying attendance records using SQLite.
"""

import sqlite3
import os
import time
from datetime import datetime

# ============ CONFIGURATION ============
DB_FOLDER = "attendance_logs"
DB_PATH = os.path.join(DB_FOLDER, "attendance.db")
COOLDOWN_SECONDS = 300  # 5 minutes — won't re-log the same person within this window


def init_db():
    """Create the attendance table if it doesn't exist."""
    os.makedirs(DB_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            date      TEXT NOT NULL,
            time      TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"📋 Attendance database ready at: {DB_PATH}")


def log_attendance(name):
    """
    Log attendance for a person.
    Returns True if logged, False if skipped due to cooldown.
    """
    now = time.time()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check cooldown — was this person logged recently?
    cursor.execute(
        "SELECT timestamp FROM attendance WHERE name = ? ORDER BY timestamp DESC LIMIT 1",
        (name,)
    )
    row = cursor.fetchone()

    if row and (now - row[0]) < COOLDOWN_SECONDS:
        conn.close()
        return False  # Cooldown active, skip

    # Log the attendance
    dt = datetime.now()
    cursor.execute(
        "INSERT INTO attendance (name, date, time, timestamp) VALUES (?, ?, ?, ?)",
        (name, dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), now)
    )
    conn.commit()
    conn.close()

    print(f"✅ Attendance logged: {name} at {dt.strftime('%H:%M:%S')}")
    return True


def get_today_attendance():
    """Get all attendance records for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    return get_attendance_by_date(today)


def get_attendance_by_date(date_str):
    """
    Get attendance records for a specific date.
    date_str format: YYYY-MM-DD
    Returns a list of dicts with 'name', 'date', 'time'.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, date, time FROM attendance WHERE date = ? ORDER BY time",
        (date_str,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [{"name": r[0], "date": r[1], "time": r[2]} for r in rows]


def get_all_dates():
    """Get all unique dates that have attendance records, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_attendance_stats():
    """Get summary stats for the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    # Today's count
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
    today_count = cursor.fetchone()[0]

    # Today's unique people
    cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance WHERE date = ?", (today,))
    today_unique = cursor.fetchone()[0]

    # Total records all time
    cursor.execute("SELECT COUNT(*) FROM attendance")
    total_records = cursor.fetchone()[0]

    # Latest check-in today
    cursor.execute(
        "SELECT name, time FROM attendance WHERE date = ? ORDER BY time DESC LIMIT 1",
        (today,)
    )
    latest = cursor.fetchone()

    conn.close()

    return {
        "today_count": today_count,
        "today_unique": today_unique,
        "total_records": total_records,
        "latest_name": latest[0] if latest else "—",
        "latest_time": latest[1] if latest else "—",
    }


def get_all_attendance():
    """Get all attendance records, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, date, time FROM attendance ORDER BY date DESC, time DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"name": r[0], "date": r[1], "time": r[2]} for r in rows]


def delete_attendance_by_name(name):
    """Delete all attendance records for a person."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE name = ?", (name,))
    conn.commit()
    conn.close()
