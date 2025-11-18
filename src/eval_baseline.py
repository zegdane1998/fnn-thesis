import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

DATA_PATH = "data/processed/politifact_articles.parquet"
MODEL_PATH = "models/baseline_tfidf_logreg.joblib"
OUT_METRICS = "metrics_baseline.json"


def load_data():
    df = pd.read_parquet(DATA_PATH)
    X = (df["title"].fillna("") + " [SEP] " + df["text"].fillna("")).astype(str)
    y = df["label"].astype(int)
    return X, y


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    X, y = load_data()
    model = joblib.load(MODEL_PATH)

    # simple hold-out for eval: last 20% as "test" (deterministic)
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": cm,
        "n_test": int(len(y_test)),
    }

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics ->", OUT_METRICS)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
