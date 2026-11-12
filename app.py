"""
Flask Application for Fake News Detection System.

Provides web routes for news analysis, interactive dashboard, prediction history,
and methodology/about pages, along with RESTful API endpoints.
All database interactions use parameterized queries via the database module.
"""

import json
import logging
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import joblib
import requests
from flask import Flask, jsonify, render_template, request

import database

# Initialize Flask application
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "fake-news-detector-dev-key-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB maximum payload

# Optional Google Fact Check Tools API Key
GOOGLE_FACTCHECK_API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "").strip()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Initialize database
database.init_db()

# Global model cache
MODEL_PIPELINE = None
MODEL_METRICS: Optional[Dict[str, Any]] = None


def load_model_pipeline() -> bool:
    """Loads the trained ML pipeline and metrics if available."""
    global MODEL_PIPELINE, MODEL_METRICS
    if os.path.exists(MODEL_PATH):
        try:
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded ML pipeline from %s", MODEL_PATH)
        except Exception as e:
            logger.error("Failed to load model pipeline: %s", e)
            MODEL_PIPELINE = None
    else:
        logger.warning("Model file not found at %s. Predictions will be unavailable until trained.", MODEL_PATH)
        MODEL_PIPELINE = None

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                MODEL_METRICS = json.load(f)
        except Exception as e:
            logger.error("Failed to load model metrics: %s", e)
            MODEL_METRICS = None

    return MODEL_PIPELINE is not None


# Initial load attempt
load_model_pipeline()

# Curated external fact check resources
FACT_CHECK_RESOURCES = [
    {
        "name": "Reuters Fact Check",
        "url": "https://www.reuters.com/fact-check/",
        "description": "Global investigations countering misinformation on major news, politics, and social media.",
        "badge": "Global Standard",
    },
    {
        "name": "Associated Press (AP) Verify",
        "url": "https://www.ap.org/verify",
        "description": "Exhaustive verification of viral news and multimedia claims by AP journalists.",
        "badge": "Independent Journalism",
    },
    {
        "name": "PIB Fact Check",
        "url": "https://factcheck.pib.gov.in/",
        "description": "Official fact-checking unit of the Press Information Bureau, Government of India.",
        "badge": "Government of India",
    },
    {
        "name": "Newschecker India",
        "url": "https://www.newschecker.in/",
        "description": "IFCN-certified fact-checking organization focusing on regional and national Indian claims.",
        "badge": "IFCN Certified",
    },
]


def sanitize_input_text(raw_text: str) -> str:
    """Strips extra whitespace while preserving readability."""
    if not raw_text:
        return ""
    # Normalize excessive newlines and spaces
    cleaned = re.sub(r"[ \t]+", " ", raw_text)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def explain_prediction(text: str, label: str, prob_real: float, prob_fake: float) -> str:
    """
    Constructs an educational explanation based on model confidence and feature presence.
    Explicitly clarifies that this reflects linguistic style matching, not real-world factual verification.
    """
    confidence = max(prob_real, prob_fake) * 100.0
    diff = abs(prob_real - prob_fake) * 100.0

    if diff < 15.0:
        return (
            f"The model detected borderline stylistic signals (Real: {prob_real*100:.1f}%, Fake: {prob_fake*100:.1f}%). "
            f"Because this excerpt contains few strongly discriminative keywords from the WELFake training corpus, "
            f"the confidence is low ({confidence:.1f}%). Submitting a longer article excerpt or cross-referencing with external fact-checkers is highly recommended."
        )

    if label == "Likely Real":
        return (
            f"The model detected writing patterns with a {confidence:.1f}% stylistic alignment with verified "
            f"news samples in the WELFake training corpus (e.g., standard neutral tone, journalistic attribution, "
            f"and structured reporting syntax). Note: This measures writing style resemblance, NOT whether the facts described are true."
        )
    else:
        return (
            f"The model detected writing characteristics with a {confidence:.1f}% stylistic alignment with unverified "
            f"or fabricated articles in the WELFake training corpus (e.g., sensational framing, lack of attribution, "
            f"or hyperbolic phrasing). Note: This does NOT prove the claim is false; always consult independent fact-checkers."
        )


def query_google_factcheck(query: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Queries official Google Fact Check Tools API for IFCN-reviewed ratings
    (Reuters Fact Check, AP Fact Check, AltNews, BoomLive, PolitiFact, Snopes, etc.).
    Requires a free Google Cloud API key.
    """
    key = (api_key or GOOGLE_FACTCHECK_API_KEY or "").strip()
    if not key:
        return []

    try:
        clean_query = " ".join(query.strip().split()[:15])
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={urllib.parse.quote(clean_query)}&key={key}&languageCode=en"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return []

        data = resp.json()
        claims = data.get("claims", [])
        results = []
        for item in claims[:3]:
            claim_text = item.get("text", "")
            claimant = item.get("claimant", "Social Media Claim")
            reviews = item.get("claimReview", [])
            if reviews:
                rev = reviews[0]
                publisher = rev.get("publisher", {}).get("name", "Verified Fact-Checker")
                rating = rev.get("textualRating", "Reviewed")
                rev_url = rev.get("url", "")
                title = rev.get("title", "")
                results.append({
                    "claim": claim_text,
                    "claimant": claimant,
                    "publisher": publisher,
                    "rating": rating,
                    "url": rev_url,
                    "title": title
                })
        return results
    except Exception as e:
        logger.warning("Google Fact Check API query failed: %s", e)
        return []


def query_live_news_wire(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """
    Queries live mainstream news wires via Google News RSS Search (no API key required).
    Returns real-time verified reports from established media houses
    (Reuters, The Hindu, NDTV, BBC, Times of India, Indian Express).
    """
    try:
        words = query.strip().split()
        clean_query = " ".join(words[:10])
        encoded = urllib.parse.quote(clean_query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FakeNewsDetector/1.0"}

        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_results]
        results = []
        for item in items:
            title_elem = item.find("title")
            link_elem = item.find("link")
            source_elem = item.find("source")
            pubdate_elem = item.find("pubDate")

            title = title_elem.text if title_elem is not None else ""
            link = link_elem.text if link_elem is not None else ""
            source = source_elem.text if source_elem is not None else "Verified Media"
            pub_date = pubdate_elem.text if pubdate_elem is not None else ""

            if title:
                results.append({
                    "title": title,
                    "link": link,
                    "source": source,
                    "pub_date": pub_date
                })
        return results
    except Exception as e:
        logger.warning("Live news wire query failed: %s", e)
        return []


# --- Web Page Routes ---


@app.route("/")
def index():
    """Renders the main news detection interface."""
    model_ready = MODEL_PIPELINE is not None
    return render_template(
        "index.html",
        model_ready=model_ready,
        fact_checkers=FACT_CHECK_RESOURCES,
    )


@app.route("/dashboard")
def dashboard():
    """Renders the interactive statistics and performance dashboard."""
    # Refresh metrics in case newly trained
    if MODEL_METRICS is None and os.path.exists(METRICS_PATH):
        load_model_pipeline()

    history_stats = database.get_history_statistics()
    return render_template(
        "dashboard.html",
        metrics=MODEL_METRICS,
        history_stats=history_stats,
        model_ready=(MODEL_PIPELINE is not None),
    )


@app.route("/history")
def history():
    """Renders the prediction audit history page."""
    records = database.get_all_predictions(limit=100)
    history_stats = database.get_history_statistics()
    return render_template(
        "history.html",
        records=records,
        stats=history_stats,
        fact_checkers=FACT_CHECK_RESOURCES,
    )


@app.route("/about")
def about():
    """Renders the methodology, dataset documentation, limitations, and ethical guidelines page."""
    return render_template(
        "about.html",
        metrics=MODEL_METRICS,
        fact_checkers=FACT_CHECK_RESOURCES,
    )


def evaluate_news_claim(cleaned_text: str, prob_real: float, prob_fake: float) -> Tuple[str, float, float, float, str]:
    """
    Evaluates news claim combining ML probabilities with linguistic attribution auditing
    to identify viral social media rumors (e.g. death hoaxes, secret arrests) lacking journalistic sourcing.
    """
    text_lower = cleaned_text.lower()
    words = text_lower.split()
    word_count = len(words)

    # Viral rumor / hoax trigger patterns (death hoaxes, secret arrests, miracle cures, declarations)
    hoax_claim_patterns = [
        r"\b(is dead|has died|died today|passed away|found dead|killed in secret|assassinated today)\b",
        r"\b(arrested today|secretly arrested|military tribunal|executed tonight)\b",
        r"\b(cures all|miracle cure|cure for cancer|secret remedy)\b",
        r"\b(world war 3 declared|martial law declared|emergency imposed tonight)\b",
    ]
    is_extreme_claim = any(re.search(pat, text_lower) for pat in hoax_claim_patterns)

    # Journalistic attribution indicators (Reuters, AP, PTI, ANI, PIB, BBC, spokesperson, etc.)
    attribution_patterns = (
        r"\b(reuters|associated press|\bap\b|pib|pti|ani|bbc|spokesperson|officials? said|"
        r"according to|police confirmed|ministry|statement|briefing|government said)\b"
    )
    has_attribution = bool(re.search(attribution_patterns, text_lower))

    # If it is an isolated extreme claim (< 30 words) with ZERO journalistic attribution,
    # it represents a classic viral social media hoax
    if is_extreme_claim and not has_attribution and word_count < 30:
        fake_score = max(prob_fake * 100.0, 88.5)
        real_score = 100.0 - fake_score
        label = "Likely Fake"
        explanation = (
            f"High Rumor Risk (Unverified Death/Arrest Claim): This statement asserts an extreme claim "
            f"in only {word_count} words with ZERO journalistic attribution, "
            "news agency, or corroborated source. Viral social media death hoaxes and disinformation take this exact form. "
            "Never trust short social media death or arrest rumors without verified reports on news wires."
        )
        return label, fake_score, fake_score, real_score, explanation

    # Standard ML confidence mapping
    if prob_real >= prob_fake:
        label = "Likely Real"
        confidence = prob_real * 100.0
        fake_prob = prob_fake * 100.0
        real_prob = prob_real * 100.0
    else:
        label = "Likely Fake"
        confidence = prob_fake * 100.0
        fake_prob = prob_fake * 100.0
        real_prob = prob_real * 100.0

    explanation = explain_prediction(cleaned_text, label, real_prob / 100.0, fake_prob / 100.0)
    return label, confidence, fake_prob, real_prob, explanation


# --- RESTful API Endpoints ---


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Evaluates submitted news text using the TF-IDF + Logistic Regression pipeline.
    Validates input bounds, logs evaluation to SQLite, and returns structured result.
    """
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        # Try reloading once in case user just ran training
        if not load_model_pipeline():
            return (
                jsonify(
                    {
                        "status": "error",
                        "error_code": "MODEL_NOT_TRAINED",
                        "message": (
                            "Machine learning model is not yet trained or loaded. "
                            "Please run 'python train_model.py' to generate the trained model artifact."
                        ),
                    }
                ),
                503,
            )

    data = request.get_json(silent=True) or request.form
    raw_text = data.get("text", "")
    source_url = data.get("source_url", "").strip()

    # Input Validation
    cleaned_text = sanitize_input_text(raw_text)

    if not cleaned_text:
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "EMPTY_INPUT",
                    "message": "Please provide a news headline or article text to analyze.",
                }
            ),
            400,
        )

    if len(cleaned_text) < 15:
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "INPUT_TOO_SHORT",
                    "message": (
                        f"Input is too short ({len(cleaned_text)} characters). "
                        "Please enter at least 15 characters (a full headline or article excerpt) for reliable feature extraction."
                    ),
                }
            ),
            400,
        )

    if len(cleaned_text) > 50000:
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "INPUT_TOO_LONG",
                    "message": (
                        f"Input exceeds the maximum allowed length of 50,000 characters "
                        f"(received {len(cleaned_text):,} characters). Please shorten the text."
                    ),
                }
            ),
            400,
        )

    # Optional URL sanity validation
    if source_url and not (source_url.startswith("http://") or source_url.startswith("https://")):
        # Prepend https if user typed www.example.com
        if source_url.startswith("www."):
            source_url = "https://" + source_url
        else:
            source_url = None

    try:
        # WELFake dataset classes: Class 0 = Real News, Class 1 = Fake News
        probabilities = MODEL_PIPELINE.predict_proba([cleaned_text])[0]
        prob_real = float(probabilities[0])
        prob_fake = float(probabilities[1])

        # Evaluates news claim combining ML probabilities with linguistic attribution auditing
        label, confidence, fake_prob, real_prob, explanation = evaluate_news_claim(
            cleaned_text, prob_real, prob_fake
        )

        # Store in SQLite database using parameterized queries
        record_id = database.insert_prediction(
            text=cleaned_text,
            label=label,
            confidence=round(confidence, 2),
            fake_prob=round(fake_prob, 2),
            real_prob=round(real_prob, 2),
            source_url=source_url,
        )

        # Query live news wires and optional Google Fact Check Tools API
        user_api_key = data.get("google_api_key", "").strip() or None
        live_fact_checks = query_google_factcheck(cleaned_text, user_api_key)
        live_news_coverage = query_live_news_wire(cleaned_text)

        return (
            jsonify(
                {
                    "status": "success",
                    "prediction_id": record_id,
                    "label": label,
                    "confidence": round(confidence, 2),
                    "probabilities": {
                        "fake": round(fake_prob, 2),
                        "real": round(real_prob, 2),
                    },
                    "explanation": explanation,
                    "live_fact_checks": live_fact_checks,
                    "live_news_coverage": live_news_coverage,
                    "has_fact_checks": len(live_fact_checks) > 0,
                    "has_live_news": len(live_news_coverage) > 0,
                    "google_factcheck_configured": bool(user_api_key or GOOGLE_FACTCHECK_API_KEY),
                    "submitted_snippet": cleaned_text[:180] + ("..." if len(cleaned_text) > 180 else ""),
                    "input_length_chars": len(cleaned_text),
                    "input_word_count": len(cleaned_text.split()),
                    "disclaimer": (
                        "This machine learning prediction reflects statistical text similarity to the "
                        "WELFake dataset. It does NOT constitute factual verification or evidence. "
                        "Always verify critical claims through independent, reputable fact-checkers."
                    ),
                    "fact_checkers": FACT_CHECK_RESOURCES,
                }
            ),
            200,
        )

    except Exception as e:
        logger.exception("Inference error occurred: %s", e)
        return (
            jsonify(
                {
                    "status": "error",
                    "error_code": "INFERENCE_ERROR",
                    "message": "An internal error occurred during model evaluation. Please try again.",
                }
            ),
            500,
        )


@app.route("/api/history", methods=["GET"])
def api_get_history():
    """Returns stored prediction history."""
    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)
    limit = min(max(1, limit), 200)
    records = database.get_all_predictions(limit=limit, offset=offset)
    return jsonify({"status": "success", "count": len(records), "records": records}), 200


@app.route("/api/history/<int:item_id>", methods=["DELETE"])
def api_delete_history_item(item_id: int):
    """Deletes a specific prediction item by ID."""
    success = database.delete_prediction(item_id)
    if success:
        return jsonify({"status": "success", "message": f"Record #{item_id} deleted."}), 200
    return jsonify({"status": "error", "message": f"Record #{item_id} not found."}), 404


@app.route("/api/history/clear", methods=["POST", "DELETE"])
def api_clear_history():
    """Clears all stored prediction records from the database."""
    deleted_count = database.clear_all_predictions()
    return (
        jsonify(
            {
                "status": "success",
                "message": f"Successfully cleared {deleted_count} prediction record(s).",
                "deleted_count": deleted_count,
            }
        ),
        200,
    )


@app.route("/api/metrics", methods=["GET"])
def api_get_metrics():
    """Returns training metrics, confusion matrix, and database history stats for dashboard."""
    if MODEL_METRICS is None and os.path.exists(METRICS_PATH):
        load_model_pipeline()

    history_stats = database.get_history_statistics()
    return (
        jsonify(
            {
                "status": "success",
                "model_loaded": MODEL_PIPELINE is not None,
                "metrics": MODEL_METRICS,
                "history_stats": history_stats,
            }
        ),
        200,
    )


@app.route("/api/status", methods=["GET"])
def api_get_status():
    """Health check endpoint indicating whether model and database are operational."""
    return (
        jsonify(
            {
                "status": "healthy",
                "model_loaded": MODEL_PIPELINE is not None,
                "model_path": MODEL_PATH,
                "metrics_available": MODEL_METRICS is not None,
            }
        ),
        200,
    )


# --- Custom Safe Error Handlers ---


@app.errorhandler(400)
def handle_bad_request(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "Bad request", "details": str(e)}), 400
    return render_template("base.html", error_title="400 - Bad Request", error_msg="The request could not be processed."), 400


@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "Requested resource was not found."}), 404
    return render_template("base.html", error_title="404 - Page Not Found", error_msg="The requested page does not exist."), 404


@app.errorhandler(500)
def handle_server_error(e):
    logger.error("Unhandled 500 error: %s", e)
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "Internal server error occurred."}), 500
    return render_template("base.html", error_title="500 - Server Error", error_msg="An unexpected error occurred. Please try again."), 500


if __name__ == "__main__":
    # Run application on localhost:5000 with debug disabled for security
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"Fake News Detection System running at http://127.0.0.1:{port}")
    print(f"Model status: {'READY' if MODEL_PIPELINE else 'NOT TRAINED (run python train_model.py)'}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
