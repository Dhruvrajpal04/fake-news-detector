"""
Automated unit tests for ML pipeline training, serialization, and inference.
"""

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_dataset():
    """Generates a small controlled benchmark dataset for pipeline verification."""
    texts = [
        "Government officials announced a new infrastructure project today in the capital city.",
        "The economic report released by the central treasury indicates stable monetary policy.",
        "Foreign ministers met for bilateral discussions regarding regional trade agreements.",
        "SHOCKING CONSPIRACY: Secret alien reptilians are controlling the city mayor and poisoned the water!",
        "MIRACLE CURE: Doctors are horrified by this 1 bizarre household trick that cures everything overnight!",
        "UNBELIEVABLE: Famous celebrity reveals hidden underground fortress before being arrested by police!",
    ]
    labels = [0, 0, 0, 1, 1, 1]  # 0 = Real, 1 = Fake
    return pd.Series(texts), pd.Series(labels)


def test_pipeline_construction_and_training(sample_dataset):
    """Verifies that the TF-IDF + Logistic Regression pipeline trains without error."""
    X, y = sample_dataset
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ("clf", LogisticRegression(random_state=42, fit_intercept=False)),
        ]
    )

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})
    assert probs.shape == (len(X), 2)
    # Check probability distribution sums to 1.0
    for prob in probs:
        assert np.isclose(np.sum(prob), 1.0)
        assert 0.0 <= prob[0] <= 1.0
        assert 0.0 <= prob[1] <= 1.0


def test_pipeline_serialization(sample_dataset):
    """Verifies that trained pipeline can be safely serialized and deserialized with joblib."""
    X, y = sample_dataset
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
            ("clf", LogisticRegression(random_state=42, fit_intercept=False)),
        ]
    )
    pipeline.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        joblib.dump(pipeline, tmp_path)
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0

        loaded_pipeline = joblib.load(tmp_path)
        test_text = ["Official statement from the government department regarding policy."]
        orig_pred = pipeline.predict(test_text)
        loaded_pred = loaded_pipeline.predict(test_text)
        assert orig_pred[0] == loaded_pred[0]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_model_pipeline_discrimination(sample_dataset):
    """Tests that model differentiates between formal report syntax and sensational keywords."""
    X, y = sample_dataset
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(random_state=42, C=10.0)),
        ]
    )
    pipeline.fit(X, y)

    real_query = ["The central bank issued the quarterly financial statement today."]
    fake_query = ["BOMBSHELL ALIEN CONSPIRACY: Secret underground bunker revealed!"]

    prob_real_for_real = pipeline.predict_proba(real_query)[0][0]
    prob_real_for_fake = pipeline.predict_proba(fake_query)[0][0]

    # Real sample should have higher probability of being real (label 0)
    assert prob_real_for_real > prob_real_for_fake
