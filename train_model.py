"""
Model Training Script for Fake News Detection System.

This script loads the official WELFake dataset, performs thorough data cleaning,
handles missing values and duplicate records, audits class distribution,
splits train and test sets to prevent data leakage, trains a TF-IDF + Logistic
Regression pipeline, evaluates performance (Accuracy, Precision, Recall, F1,
Confusion Matrix), extracts top linguistic feature coefficients, and persists
the pipeline artifact and metrics JSON.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "fake_news_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")


def check_dataset_exists() -> bool:
    """Checks if the dataset exists and is non-empty."""
    return os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 100_000


def clean_text_for_training(text: str) -> str:
    """Basic preprocessing: clean extra whitespaces and standardize strings."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def load_and_preprocess_data(max_samples: int = None) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Loads WELFake dataset, cleans missing values and duplicates,
    combines title + body text, and reports dataset statistics.
    WELFake Labels:
      0 = Fake
      1 = Real
    """
    if not check_dataset_exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Please run 'python download_dataset.py' to download the official WELFake dataset (245 MB) from Zenodo."
        )

    print("=" * 70)
    print("STEP 1: Loading and inspecting dataset...")
    print(f"Reading from: {DATASET_PATH}")
    start_time = time.time()

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    raw_count = len(df)
    print(f"Raw rows loaded: {raw_count:,}")
    print(f"Columns found: {list(df.columns)}")

    # Standardize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns check
    if "label" not in df.columns:
        raise ValueError(f"Expected 'label' column in dataset, but found: {list(df.columns)}")

    # Record missing values before cleaning
    missing_title = int(df["title"].isna().sum()) if "title" in df.columns else 0
    missing_text = int(df["text"].isna().sum()) if "text" in df.columns else 0
    missing_label = int(df["label"].isna().sum())

    print(f"Missing values before cleaning - Title: {missing_title:,}, Text: {missing_text:,}, Label: {missing_label:,}")

    # Drop rows where label is missing or non-binary
    df = df.dropna(subset=["label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    # Fill null text/title with empty string
    title_series = df["title"].fillna("").astype(str) if "title" in df.columns else pd.Series([""] * len(df))
    text_series = df["text"].fillna("").astype(str) if "text" in df.columns else pd.Series([""] * len(df))

    # Combine title and text: "Title. Text"
    print("Combining headline/title and body text...")
    df["content"] = title_series.str.strip() + ". " + text_series.str.strip()
    df["content"] = df["content"].str.strip()

    # Drop rows where content is empty or contains only punctuation/spaces
    df = df[df["content"].str.len() > 10]

    # Deduplication
    duplicates_count = int(df.duplicated(subset=["content"]).sum())
    print(f"Duplicate content records identified: {duplicates_count:,}. Dropping duplicates...")
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)

    cleaned_count = len(df)
    print(f"Cleaned valid records: {cleaned_count:,} (Removed {raw_count - cleaned_count:,} invalid/duplicate rows)")

    # Class distribution in WELFake: 0 = Real News, 1 = Fake News
    label_counts = df["label"].value_counts().to_dict()
    real_count = int(label_counts.get(0, 0))
    fake_count = int(label_counts.get(1, 0))
    print(f"Class distribution: Real (0) = {real_count:,} ({real_count/cleaned_count*100:.1f}%), Fake (1) = {fake_count:,} ({fake_count/cleaned_count*100:.1f}%)")

    # Optional subsampling for quick validation/prototyping
    if max_samples and max_samples < cleaned_count:
        print(f"Subsampling to {max_samples:,} records for accelerated training...")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    # Augmented dataset combining full article bodies and standalone headlines
    # This ensures the model learns stylistic signals for both short headlines and long articles,
    # preventing the common failure mode where short inputs default to 'Fake'
    titles_df = df[df["title"].fillna("").astype(str).str.strip().str.len() > 10].copy()
    X_all = pd.concat([df["content"], titles_df["title"].astype(str).str.strip()], ignore_index=True)
    y_all = pd.concat([df["label"], titles_df["label"]], ignore_index=True)
    print(f"Combined evaluation samples: {len(X_all):,} ({len(df):,} full articles + {len(titles_df):,} headlines)")

    stats = {
        "raw_rows": raw_count,
        "cleaned_rows": len(df),
        "total_training_samples": len(X_all),
        "removed_rows": raw_count - len(df),
        "duplicates_removed": duplicates_count,
        "class_distribution": {
            "real_count": int((y_all == 0).sum()),
            "fake_count": int((y_all == 1).sum()),
        },
        "load_time_seconds": round(time.time() - start_time, 2),
    }

    return X_all, y_all, stats


def train_and_evaluate(
    X: pd.Series,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """
    Splits data into train and test sets to prevent data leakage,
    builds a TF-IDF + Logistic Regression pipeline, fits strictly on training fold,
    and evaluates on test fold.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Splitting dataset into Train and Test folds...")
    print(f"Test size: {test_size * 100:.0f}%, Stratified by label to maintain class balance")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")

    print("\n" + "=" * 70)
    print("STEP 3: Building and training TF-IDF + Logistic Regression pipeline...")
    print("Vectorizer: TF-IDF (1-gram & 2-gram, max 50,000 features, English stop words, sublinear TF)")
    print("Classifier: Logistic Regression (L2 regularization, solver='lbfgs', max_iter=1000)")

    train_start = time.time()

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50000,
                    stop_words="english",
                    sublinear_tf=True,
                    lowercase=True,
                    min_df=3,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    fit_intercept=False,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # Strictly fit pipeline on training data to prevent any data leakage
    pipeline.fit(X_train, y_train)
    training_duration = round(time.time() - train_start, 2)
    print(f"Training completed in {training_duration} seconds.")

    print("\n" + "=" * 70)
    print("STEP 4: Evaluating model on unseen test set...")
    eval_start = time.time()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # Performance metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro"))
    rec_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    # In WELFake dataset: Label 0 = Real News, Label 1 = Fake News
    prec_real = float(precision_score(y_test, y_pred, pos_label=0))
    rec_real = float(recall_score(y_test, y_pred, pos_label=0))
    f1_real = float(f1_score(y_test, y_pred, pos_label=0))

    prec_fake = float(precision_score(y_test, y_pred, pos_label=1))
    rec_fake = float(recall_score(y_test, y_pred, pos_label=1))
    f1_fake = float(f1_score(y_test, y_pred, pos_label=1))

    # Confusion Matrix:
    # row 0 = actual real (0), row 1 = actual fake (1)
    # col 0 = pred real (0),   col 1 = pred fake (1)
    cm = confusion_matrix(y_test, y_pred)
    true_real = int(cm[0, 0])
    false_fake = int(cm[0, 1])
    false_real = int(cm[1, 0])
    true_fake = int(cm[1, 1])

    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro Precision:   {prec_macro * 100:.2f}%")
    print(f"Macro Recall:      {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%")
    print("\nPer-Class Breakdown:")
    print(f"  [Real (0)] Precision: {prec_real*100:.2f}%, Recall: {rec_real*100:.2f}%, F1: {f1_real*100:.2f}%")
    print(f"  [Fake (1)] Precision: {prec_fake*100:.2f}%, Recall: {rec_fake*100:.2f}%, F1: {f1_fake*100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  True Real (0,0):  {true_real:,} | False Fake (0,1): {false_fake:,}")
    print(f"  False Real (1,0): {false_real:,} | True Fake (1,1):  {true_fake:,}")

    # Extract top linguistic features by coefficient magnitude
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]

    # In binary Logistic Regression with class 0 = Real, class 1 = Fake:
    # Positive weights push toward Fake (class 1)
    # Negative weights push toward Real (class 0)
    top_n = 15
    top_fake_idx = np.argsort(coefs)[-top_n:][::-1]
    top_real_idx = np.argsort(coefs)[:top_n]

    top_fake_features = [
        {"feature": feature_names[i], "weight": round(float(coefs[i]), 4)} for i in top_fake_idx
    ]
    top_real_features = [
        {"feature": feature_names[i], "weight": round(float(abs(coefs[i])), 4)} for i in top_real_idx
    ]

    metrics = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model_architecture": "TF-IDF (unigrams+bigrams) + Logistic Regression (L2)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": len(feature_names),
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "class_metrics": {
            "real": {
                "label": "Real (0)",
                "precision": round(prec_real, 4),
                "recall": round(rec_real, 4),
                "f1_score": round(f1_real, 4),
                "support": int((y_test == 0).sum()),
            },
            "fake": {
                "label": "Fake (1)",
                "precision": round(prec_fake, 4),
                "recall": round(rec_fake, 4),
                "f1_score": round(f1_fake, 4),
                "support": int((y_test == 1).sum()),
            },
        },
        "confusion_matrix": {
            "true_real": true_real,
            "false_fake": false_fake,
            "false_real": false_real,
            "true_fake": true_fake,
            "matrix": [[true_real, false_fake], [false_real, true_fake]],
        },
        "top_features": {
            "indicative_of_real": top_real_features,
            "indicative_of_fake": top_fake_features,
        },
        "training_time_seconds": training_duration,
        "evaluation_time_seconds": round(time.time() - eval_start, 2),
    }

    return pipeline, metrics


def save_artifacts(pipeline: Pipeline, metrics: Dict) -> None:
    """Saves the trained pipeline and evaluation metrics to disk."""
    print("\n" + "=" * 70)
    print("STEP 5: Persisting model artifact and metrics...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[OK] Model pipeline saved to: {MODEL_PATH} ({file_size_mb:.2f} MB)")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Evaluation metrics saved to: {METRICS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train Fake News Detection Model using WELFake dataset")
    parser.add_argument("--download", action="store_true", help="Download WELFake dataset if not found")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples for fast testing")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set fraction (default: 0.20)")
    args = parser.parse_args()

    if not check_dataset_exists():
        if args.download:
            from download_dataset import download_welfake_dataset
            download_welfake_dataset()
        else:
            print("=" * 70)
            print("[ERROR] WELFake dataset file not found at:")
            print(f"  {DATASET_PATH}")
            print("\nPlease either:")
            print("  1. Run 'python download_dataset.py' to download the 245 MB dataset automatically.")
            print("  2. Run 'python train_model.py --download'")
            print("  3. Manually download 'WELFake_Dataset.csv' from https://zenodo.org/records/4561253")
            print("     and place it into the 'data/' directory.")
            print("=" * 70)
            sys.exit(1)

    X, y, data_stats = load_and_preprocess_data(max_samples=args.max_samples)
    pipeline, metrics = train_and_evaluate(X, y, test_size=args.test_size)
    metrics["dataset_statistics"] = data_stats
    save_artifacts(pipeline, metrics)
    print("\n[SUCCESS] Model training and evaluation successfully completed!")


if __name__ == "__main__":
    main()
