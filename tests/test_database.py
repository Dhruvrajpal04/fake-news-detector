"""
Automated unit tests for SQLite database operations.
"""

import os
import tempfile
import pytest
import database


@pytest.fixture
def temp_db():
    """Provides an isolated temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database.init_db(db_path=path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_init_db(temp_db):
    """Verifies database tables are created correctly."""
    conn = database.get_db_connection(temp_db)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';"
    )
    table = cursor.fetchone()
    conn.close()
    assert table is not None
    assert table["name"] == "predictions"


def test_insert_and_get_prediction(temp_db):
    """Tests parameterized record insertion and retrieval."""
    test_text = "WASHINGTON - The Senate committee passed the defense appropriation bill on Wednesday."
    record_id = database.insert_prediction(
        text=test_text,
        label="Likely Real",
        confidence=94.5,
        fake_prob=5.5,
        real_prob=94.5,
        source_url="https://example.com/news",
        db_path=temp_db,
    )
    assert record_id > 0

    record = database.get_prediction_by_id(record_id, db_path=temp_db)
    assert record is not None
    assert record["id"] == record_id
    assert record["prediction_label"] == "Likely Real"
    assert record["confidence"] == 94.5
    assert record["fake_probability"] == 5.5
    assert record["real_probability"] == 94.5
    assert record["source_url"] == "https://example.com/news"
    assert record["full_text"] == test_text


def test_get_history_statistics(temp_db):
    """Tests aggregate history analytics."""
    database.insert_prediction("Text 1", "Likely Real", 90.0, 10.0, 90.0, db_path=temp_db)
    database.insert_prediction("Text 2", "Likely Fake", 80.0, 80.0, 20.0, db_path=temp_db)
    database.insert_prediction("Text 3", "Likely Real", 70.0, 30.0, 70.0, db_path=temp_db)

    stats = database.get_history_statistics(db_path=temp_db)
    assert stats["total_queries"] == 3
    assert stats["real_detected"] == 2
    assert stats["fake_detected"] == 1
    assert stats["average_confidence"] == pytest.approx(80.0, 0.1)


def test_delete_and_clear_predictions(temp_db):
    """Tests individual deletion and complete history purge."""
    id1 = database.insert_prediction("Text 1", "Likely Real", 90.0, 10.0, 90.0, db_path=temp_db)
    id2 = database.insert_prediction("Text 2", "Likely Fake", 80.0, 80.0, 20.0, db_path=temp_db)

    # Delete single item
    deleted = database.delete_prediction(id1, db_path=temp_db)
    assert deleted is True
    assert database.get_prediction_by_id(id1, db_path=temp_db) is None
    assert database.get_prediction_by_id(id2, db_path=temp_db) is not None

    # Clear all
    cleared = database.clear_all_predictions(db_path=temp_db)
    assert cleared == 1
    assert len(database.get_all_predictions(db_path=temp_db)) == 0
