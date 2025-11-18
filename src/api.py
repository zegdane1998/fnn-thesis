import os
from typing import List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/baseline_tfidf_logreg.joblib"

app = FastAPI(title="Fake News Detection API (Baseline)")

model = None


class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponseItem(BaseModel):
    text: str
    label: int
    label_name: str
    prob_fake: float
    prob_real: float


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    load_model()
    probs = model.predict_proba(req.texts)
    labels = model.predict(req.texts)
    out = []
    for text, label, p in zip(req.texts, labels, probs):
        prob_fake = float(p[0])
        prob_real = float(p[1])
        label_name = "real" if label == 1 else "fake"
        out.append(
            PredictResponseItem(
                text=text,
                label=int(label),
                label_name=label_name,
                prob_fake=prob_fake,
                prob_real=prob_real,
            )
        )
    return out
