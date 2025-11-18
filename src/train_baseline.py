import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/processed/politifact_articles.parquet"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    df = pd.read_parquet(DATA_PATH)

    # text = title + [SEP] + body
    X = (df["title"].fillna("") + " [SEP] " + df["text"].fillna("")).astype(str)
    y = df["label"].astype(int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    solver="liblinear",
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    print("=== Validation ===")
    y_pred_val = pipe.predict(X_val)
    print(classification_report(y_val, y_pred_val, digits=3))
    print("Confusion matrix (val):")
    print(confusion_matrix(y_val, y_pred_val))

    print("\n=== Test ===")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred))

    out_path = os.path.join(MODEL_DIR, "baseline_tfidf_logreg.joblib")
    joblib.dump(pipe, out_path)
    print("\nSaved model ->", out_path)


if __name__ == "__main__":
    main()
