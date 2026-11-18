from pathlib import Path
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "air_quality_model.pkl"

app = FastAPI(
    title="Air Quality RH Prediction API",
    description="Predicts Relative Humidity (RH) from UCI Air Quality sensor measurements.",
    version="1.0.0",
)

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model file is missing. Run `python train_model.py` before starting the API."
    )

with MODEL_PATH.open("rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
features = bundle["features"]

class AirQualityInput(BaseModel):
    values: dict[str, float] = Field(
        ..., description="Input feature names and numeric measurements."
    )

@app.get("/")
def home():
    return {
        "message": "Air Quality RH Prediction API is running",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
        "required_features": features,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(payload: AirQualityInput):
    missing = [name for name in features if name not in payload.values]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})
    try:
        row = pd.DataFrame([[payload.values[name] for name in features]], columns=features)
        transformed = scaler.transform(imputer.transform(row))
        result = float(model.predict(transformed)[0])
        return {"predicted_RH": result, "unit": "%"}
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
