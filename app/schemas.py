"""Request and response models for the heart disease API."""

from pydantic import BaseModel, ConfigDict, Field

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class HeartFeatures(BaseModel):
    """Clinical features from the Heart Disease dataset."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0–3)")
    trestbps: int = Field(..., ge=1, le=300, description="Resting blood pressure in mm Hg")
    chol: int = Field(..., ge=1, le=700, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0–2)")
    thalach: int = Field(..., ge=1, le=300, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0–2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0–4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia status (0–3)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "age": 52,
                    "sex": 1,
                    "cp": 0,
                    "trestbps": 125,
                    "chol": 212,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 168,
                    "exang": 0,
                    "oldpeak": 1.0,
                    "slope": 2,
                    "ca": 2,
                    "thal": 3,
                }
            ]
        }
    )


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    model_type: str
    features: list[str]
    target: str
    dataset: str


class PredictionResponse(BaseModel):
    heart_disease: bool
