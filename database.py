"""
Database management module for the Fake News Detection System.
Provides SQLite database initialization, parameterized CRUD operations,
and statistical aggregation for user history and dashboard analytics.
"""

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "predictions.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database with Row factory enabled."""
    target_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initializes the database schema if tables do not already exist."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_snippet TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    prediction_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fake_probability REAL NOT NULL,
                    real_probability REAL NOT NULL,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at
                ON predictions (created_at DESC);
                """
            )
    finally:
        conn.close()


def insert_prediction(
    text: str,
    label: str,
    confidence: float,
    fake_prob: float,
    real_prob: float,
    source_url: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Inserts a newly evaluated prediction using parameterized queries.
    Saves a preview snippet (first 180 characters) as well as the sanitized text.
    """
    text_clean = text.strip()
    snippet = text_clean[:180] + ("..." if len(text_clean) > 180 else "")
    source = (source_url or "").strip()[:500] if source_url else None

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    text_snippet,
                    full_text,
                    prediction_label,
                    confidence,
                    fake_probability,
                    real_probability,
                    source_url,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    snippet,
                    text_clean,
                    label,
                    float(confidence),
                    float(fake_prob),
                    float(real_prob),
                    source,
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            return int(cursor.lastrowid)
    finally:
        conn.close()


def get_all_predictions(
    limit: int = 100,
    offset: int = 0,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves prediction history records in descending chronological order."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, text_snippet, full_text, prediction_label,
                   confidence, fake_probability, real_probability,
                   source_url, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_prediction_by_id(prediction_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetches a single prediction record by primary key."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, text_snippet, full_text, prediction_label,
                   confidence, fake_probability, real_probability,
                   source_url, created_at
            FROM predictions
            WHERE id = ?;
            """,
            (prediction_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_prediction(prediction_id: int, db_path: Optional[str] = None) -> bool:
    """Deletes a specific prediction entry by ID."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM predictions WHERE id = ?;", (prediction_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_all_predictions(db_path: Optional[str] = None) -> int:
    """Clears all stored prediction records from the database."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM predictions;")
            # Vacuum can be skipped or run in separate transaction if desired
            return cursor.rowcount
    finally:
        conn.close()


def get_history_statistics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Computes aggregate analytics over the user's prediction history."""
    conn = get_db_connection(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
        fake_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction_label = 'Likely Fake';"
        ).fetchone()[0]
        real_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction_label = 'Likely Real';"
        ).fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions;").fetchone()[0] or 0.0

        return {
            "total_queries": int(total),
            "fake_detected": int(fake_count),
            "real_detected": int(real_count),
            "average_confidence": round(float(avg_conf), 2),
        }
    finally:
        conn.close()
