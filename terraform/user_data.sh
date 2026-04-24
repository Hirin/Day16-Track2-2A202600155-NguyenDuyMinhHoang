#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting CPU ML Inference Node Setup ==="

# Update system and install dependencies
apt-get update -y
apt-get install -y python3 python3-pip python3-venv

# Create application directory
mkdir -p /opt/ml-app
cd /opt/ml-app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install ML and API packages
pip install --upgrade pip
pip install fastapi uvicorn lightgbm scikit-learn pandas numpy

# Create the ML inference API server
cat > /opt/ml-app/app.py << 'PYEOF'
import json
import time
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="ML Inference API (CPU - LightGBM)",
    description="Credit Card Fraud Detection using LightGBM on CPU instance",
    version="1.0.0"
)

# Global model reference
model = None
feature_names = None

class PredictRequest(BaseModel):
    features: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    inference_time_ms: float

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "instance_type": "cpu",
        "framework": "lightgbm"
    }

@app.get("/docs_info")
def docs_info():
    return {
        "endpoints": ["/health", "/predict", "/model/info"],
        "description": "LightGBM Credit Card Fraud Detection API"
    }

@app.get("/model/info")
def model_info():
    if model is None:
        return {"status": "no model loaded yet — run benchmark.py first"}
    return {
        "model_type": "LightGBM",
        "task": "Credit Card Fraud Detection",
        "features": feature_names if feature_names else "unknown",
        "num_features": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run benchmark.py first.")
    
    start = time.time()
    X = np.array(request.features)
    proba = model.predict_proba(X)
    preds = (proba[:, 1] >= 0.5).astype(int).tolist()
    elapsed_ms = (time.time() - start) * 1000
    
    return PredictResponse(
        predictions=preds,
        probabilities=proba.tolist(),
        inference_time_ms=round(elapsed_ms, 3)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
PYEOF

# Create systemd service for the API
cat > /etc/systemd/system/ml-api.service << 'SVCEOF'
[Unit]
Description=ML Inference API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ml-app
ExecStart=/opt/ml-app/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

# Start the API server
systemctl daemon-reload
systemctl enable ml-api
systemctl start ml-api

echo "=== CPU ML Inference Node Setup Complete ==="
echo "API server running on port 8000"
echo "Run benchmark.py manually after SSH to train model and load it"