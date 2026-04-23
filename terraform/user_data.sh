#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting user_data setup for ML Inference Endpoint ==="

# Force apt to use IPv4 only (NAT Gateway only supports IPv4)
echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

# Install Docker via official script (more reliable than docker.io package)
echo "[1/4] Installing Docker..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io
systemctl enable docker
systemctl start docker
echo "[1/4] Docker installed successfully."

# Create ML app directory
echo "[2/4] Creating ML application files..."
mkdir -p /opt/ml-app

cat > /opt/ml-app/requirements.txt << 'REQEOF'
scikit-learn==1.5.2
fastapi==0.115.6
uvicorn[standard]==0.34.0
joblib==1.4.2
numpy==1.26.4
pydantic==2.10.3
REQEOF

cat > /opt/ml-app/train_model.py << 'TRAINEOF'
import os, json, numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def train_and_save():
    print("Training House Price Prediction model...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/house_price_model.joblib")
    metadata = {
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 200,
        "feature_names": feature_names,
        "feature_descriptions": {
            "MedInc": "Median income in block group",
            "HouseAge": "Median house age in block group",
            "AveRooms": "Average number of rooms per household",
            "AveBedrms": "Average number of bedrooms per household",
            "Population": "Block group population",
            "AveOccup": "Average number of household members",
            "Latitude": "Block group latitude",
            "Longitude": "Block group longitude",
        },
        "target_description": "Median house value in $100,000s",
        "metrics": {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)},
        "dataset": "California Housing (sklearn built-in)",
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }
    with open("artifacts/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model trained! MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    train_and_save()
TRAINEOF

cat > /opt/ml-app/app.py << 'APPEOF'
import json, numpy as np, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "artifacts/house_price_model.joblib"
METADATA_PATH = "artifacts/model_metadata.json"
model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)
FEATURE_NAMES = metadata["feature_names"]
app = FastAPI(title="House Price Prediction API", version="1.0.0")

class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

class PredictionResponse(BaseModel):
    predicted_price_100k: float
    predicted_price_usd: str
    model_type: str
    features_received: dict

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_type": metadata.get("model_type"), "metrics": metadata.get("metrics")}

@app.get("/")
def root():
    return {"service": "House Price Prediction API", "version": "1.0.0", "endpoints": {"/health": "Health check", "/predict": "POST - predict house price", "/model-info": "GET - model metadata", "/docs": "Swagger UI"}}

@app.get("/model-info")
def model_info():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        feature_values = [features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms, features.Population, features.AveOccup, features.Latitude, features.Longitude]
        X = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(X)[0]
        price_usd = prediction * 100_000
        return PredictionResponse(predicted_price_100k=round(float(prediction), 4), predicted_price_usd="${price_usd:,.0f}".format(price_usd=price_usd), model_type=metadata["model_type"], features_received=features.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
APPEOF

cat > /opt/ml-app/Dockerfile << 'DKEOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_model.py .
COPY app.py .
RUN python train_model.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
DKEOF

echo "[2/4] ML application files created."

# Build Docker image
echo "[3/4] Building Docker image (this may take 2-3 minutes)..."
cd /opt/ml-app
docker build -t house-price-api:latest .
echo "[3/4] Docker image built successfully."

# Run container
echo "[4/4] Starting ML API container..."
docker run -d --name ml-api \
  --restart unless-stopped \
  -p 8000:8000 \
  house-price-api:latest

echo "=== ML API container started — House Price Prediction endpoint ready on port 8000 ==="