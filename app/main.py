"""FastAPI service that serves the trained heart disease classifier."""

from pathlib import Path

import joblib
from fastapi import FastAPI

from app.schemas import FEATURES, HealthResponse, HeartFeatures, InfoResponse, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
model_type = bundle["model_type"]
feature_names = bundle.get("features", FEATURES)

app = FastAPI(
    title="Heart Disease Classifier",
    description="Predicts presence or absence of heart disease from the UCI/Kaggle Heart Disease features.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse, tags=["meta"])
def info() -> InfoResponse:
    return InfoResponse(
        model_type=model_type,
        features=list(feature_names),
        target="heart_disease (true = presence, false = absence)",
        dataset="Heart Disease Dataset (Kaggle: johnsmith88/heart-disease-dataset)",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: HeartFeatures) -> PredictionResponse:
    row = [[getattr(payload, name) for name in feature_names]]
    prediction = int(model.predict(row)[0])
    return PredictionResponse(heart_disease=bool(prediction))
