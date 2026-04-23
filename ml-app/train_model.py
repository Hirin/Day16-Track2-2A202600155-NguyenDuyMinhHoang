"""
Train a House Price Prediction model using California Housing dataset.
Exports the trained model as a joblib file for serving via FastAPI.
"""

import os
import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def train_and_save():
    print("=" * 60)
    print("  House Price Prediction — Model Training")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/5] Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    print(f"  Dataset shape: {X.shape}")
    print(f"  Features: {feature_names}")
    print(f"  Target: Median house value (in $100,000s)")

    # 2. Split data
    print("\n[2/5] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    # 3. Build pipeline (StandardScaler + GradientBoostingRegressor)
    print("\n[3/5] Training GradientBoostingRegressor pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("  Training complete!")

    # 4. Evaluate
    print("\n[4/5] Evaluating model on test set...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE  : {mae:.4f}  (~${mae * 100_000:,.0f})")
    print(f"  RMSE : {rmse:.4f}  (~${rmse * 100_000:,.0f})")
    print(f"  R²   : {r2:.4f}")

    # 5. Save artifacts
    print("\n[5/5] Saving model artifacts...")
    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/house_price_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"  Model saved to: {model_path}")

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
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        },
        "dataset": "California Housing (sklearn built-in)",
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }

    metadata_path = "artifacts/model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path}")

    print("\n" + "=" * 60)
    print("  Training complete! Model ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
