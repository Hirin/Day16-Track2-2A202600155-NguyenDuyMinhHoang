"""
FastAPI application for serving the House Price Prediction model.
Provides /predict and /health endpoints.
"""

import json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ---------------------------------------------------------------------------
# Load model & metadata at startup
# ---------------------------------------------------------------------------
MODEL_PATH = "artifacts/house_price_model.joblib"
METADATA_PATH = "artifacts/model_metadata.json"

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["feature_names"]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",
    description="Predict California house prices using a GradientBoosting model.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HouseFeatures(BaseModel):
    """Input features for a single house price prediction."""
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.984,
                    "AveBedrms": 1.024,
                    "Population": 322.0,
                    "AveOccup": 2.556,
                    "Latitude": 37.88,
                    "Longitude": -122.23,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_100k: float = Field(
        ..., description="Predicted median house value in $100,000s"
    )
    predicted_price_usd: str = Field(
        ..., description="Predicted median house value in USD (formatted)"
    )
    model_type: str
    features_received: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint for ALB."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": metadata.get("model_type"),
        "metrics": metadata.get("metrics"),
    }


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "service": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "POST — predict house price",
            "/model-info": "GET — model metadata",
            "/docs": "Swagger UI documentation",
        },
    }


@app.get("/model-info")
def model_info():
    """Return model metadata and feature descriptions."""
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """Predict the median house value given input features."""
    try:
        # Build feature array in the correct order
        feature_values = [
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude,
        ]

        X = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(X)[0]
        price_usd = prediction * 100_000

        return PredictionResponse(
            predicted_price_100k=round(float(prediction), 4),
            predicted_price_usd=f"${price_usd:,.0f}",
            model_type=metadata["model_type"],
            features_received=features.model_dump(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
