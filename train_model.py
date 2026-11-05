"""Train a logistic regression classifier and save it with joblib.

Dataset: Kaggle Heart Disease Dataset (johnsmith88/heart-disease-dataset),
a 14-column subset of the UCI Heart Disease collection. The copy in
data/heart.csv was downloaded from a public GitHub mirror of that file.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.schemas import FEATURES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_PATH = ROOT / "model" / "heart_model.joblib"


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    missing = [name for name in FEATURES + ["target"] if name not in frame.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns: {missing}")

    x = frame[FEATURES]
    y = frame["target"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, pipeline.predict(x_test))
    print(f"Holdout accuracy: {accuracy:.3f} ({len(frame)} rows)")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "model_type": "LogisticRegression",
            "features": FEATURES,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
