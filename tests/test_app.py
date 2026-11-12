"""
Automated unit and integration tests for Flask web endpoints and REST API.
"""

import os
import tempfile
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import app as flask_app
import database


@pytest.fixture
def client():
    """Sets up a test Flask application client with an isolated temporary SQLite database."""
    flask_app.app.config["TESTING"] = True
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    # Point database module to temporary database
    database.init_db(db_path=temp_db_path)
    old_default_db = database.DEFAULT_DB_PATH
    database.DEFAULT_DB_PATH = temp_db_path

    # Ensure a small mock model is loaded for prediction tests if none exists
    if flask_app.MODEL_PIPELINE is None:
        mock_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("clf", LogisticRegression(fit_intercept=False)),
            ]
        )
        mock_pipeline.fit(
            [
                "The government official reported on standard budget allocations.",
                "SHOCKING CONSPIRACY: Secret aliens took over the city hall overnight!",
            ],
            [0, 1],
        )
        flask_app.MODEL_PIPELINE = mock_pipeline

    with flask_app.app.test_client() as test_client:
        yield test_client

    database.DEFAULT_DB_PATH = old_default_db
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)


def test_home_page(client):
    """Verifies that the home detector page renders correctly with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"FakeNewsLens" in response.data
    assert b"Analyze News" in response.data


def test_dashboard_page(client):
    """Verifies that the dashboard page renders with HTTP 200."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Model Performance &amp; Analytics Dashboard" in response.data or b"Dashboard" in response.data


def test_history_page(client):
    """Verifies that the history audit page renders with HTTP 200."""
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Prediction Audit History" in response.data


def test_about_page(client):
    """Verifies that the about and methodology page renders with HTTP 200."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"WELFake" in response.data
    assert b"TF-IDF" in response.data


def test_api_status(client):
    """Verifies the health check API endpoint."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_api_predict_empty_input(client):
    """Verifies that empty input yields a 400 Bad Request."""
    response = client.post("/api/predict", json={"text": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_code"] == "EMPTY_INPUT"


def test_api_predict_short_input(client):
    """Verifies that excessively short input yields a 400 Bad Request."""
    response = client.post("/api/predict", json={"text": "Hello news"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_code"] == "INPUT_TOO_SHORT"


def test_api_predict_excessive_input(client):
    """Verifies that input exceeding 50,000 characters is rejected."""
    long_text = "word " * 15000  # > 75,000 chars
    response = client.post("/api/predict", json={"text": long_text})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error_code"] == "INPUT_TOO_LONG"


def test_api_predict_valid_input(client):
    """Verifies end-to-end inference flow returns compliant prediction structure."""
    valid_text = (
        "WASHINGTON (Reuters) - Federal Reserve policymakers met today to review current economic "
        "growth benchmarks and signaled that monetary policy adjustments remain data-dependent."
    )
    response = client.post(
        "/api/predict",
        json={"text": valid_text, "source_url": "https://reuters.com/article"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["label"] in ["Likely Real", "Likely Fake"]
    assert "confidence" in data
    assert "probabilities" in data
    assert "real" in data["probabilities"]
    assert "fake" in data["probabilities"]
    assert "disclaimer" in data
    assert "fact_checkers" in data
    assert len(data["fact_checkers"]) >= 4

    # Verify that prediction was recorded in history
    history_res = client.get("/api/history")
    assert history_res.status_code == 200
    history_data = history_res.get_json()
    assert history_data["count"] >= 1


def test_api_clear_history(client):
    """Verifies the API endpoint for clearing prediction history."""
    # Seed an entry
    client.post(
        "/api/predict",
        json={"text": "This is a legitimate news story regarding annual fiscal policy updates."},
    )
    # Clear
    clear_res = client.post("/api/history/clear")
    assert clear_res.status_code == 200
    data = clear_res.get_json()
    assert data["status"] == "success"

    # Verify history is empty
    history_res = client.get("/api/history")
    assert history_res.get_json()["count"] == 0


def test_custom_404_handler(client):
    """Verifies that non-existent API routes return clean JSON error instead of stack traces."""
    response = client.get("/api/non-existent-route")
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"
