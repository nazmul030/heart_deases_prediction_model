# Heart Disease Classifier API

FastAPI service that predicts presence or absence of heart disease from the [Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). A logistic regression model is saved with joblib and served from a Docker image.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/info` | Model type and feature list |
| POST | `/predict` | Returns `{ "heart_disease": true/false }` |
| GET | `/docs` | Swagger UI |

## Train the model

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

This writes `model/heart_model.joblib`. The committed model was trained this way, so you can skip training and go straight to Docker.

## Run with Docker

```bash
docker compose build
docker compose up
```

Swagger UI: http://localhost:8000/docs

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/info
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```

The sample body above is a row from the dataset labeled absence, so the model returns `heart_disease: false`.

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, create a **Web Service** and connect the repo.
3. Set **Language** to **Docker**. The Dockerfile is at the repository root, so the build context is the root.
4. Render injects `PORT`. The container listens on `0.0.0.0` and that port.
5. After the deploy finishes, call `/health`, `/info`, and `/predict` on the public URL.

`render.yaml` is a Render blueprint for the same service (`healthCheckPath: /health`).
