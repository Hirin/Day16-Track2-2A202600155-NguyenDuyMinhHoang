#!/usr/bin/env python3
"""
Benchmark script for LightGBM Credit Card Fraud Detection
Lab 16 - CPU Fallback Method (Phần 7 README_aws.md)

Dataset: Credit Card Fraud Detection (Kaggle - mlg-ulb/creditcardfraud)
Model: LightGBM (Gradient Boosting)

Usage:
    python3 benchmark.py

Requirements:
    pip3 install lightgbm scikit-learn pandas numpy
"""

import time
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.expanduser("~/ml-benchmark")
DATA_FILE = os.path.join(DATA_DIR, "creditcard.csv")
RESULT_FILE = os.path.join(DATA_DIR, "benchmark_result.json")
MODEL_FILE = os.path.join(DATA_DIR, "lgbm_model.pkl")


def load_data():
    """Load Credit Card Fraud Detection dataset"""
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Dataset not found at {DATA_FILE}")
        print("Please download it first:")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/")
        sys.exit(1)

    start = time.time()
    df = pd.read_csv(DATA_FILE)
    load_time = time.time() - start

    print(f"  Dataset shape: {df.shape}")
    print(f"  Fraud ratio: {df['Class'].mean():.4%}")
    print(f"  Load time: {load_time:.3f}s")
    print()

    return df, load_time


def train_model(df):
    """Train LightGBM model with cross-validation"""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        precision_score, recall_score, classification_report
    )

    print("=" * 60)
    print("STEP 2: Training LightGBM Model")
    print("=" * 60)

    # Prepare features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")
    print(f"  Fraud in train: {y_train.sum()} ({y_train.mean():.4%})")
    print(f"  Fraud in test:  {y_test.sum()} ({y_test.mean():.4%})")
    print()

    # LightGBM parameters optimized for fraud detection
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.01,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "scale_pos_weight": 50,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Train with early stopping
    print("  Training LightGBM...")
    start = time.time()

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50)
    ]

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    train_time = time.time() - start
    best_iteration = gbm.best_iteration

    print(f"\n  Training time: {train_time:.3f}s")
    print(f"  Best iteration: {best_iteration}")
    print()

    # Evaluate on test set
    print("=" * 60)
    print("STEP 3: Evaluation Metrics")
    print("=" * 60)

    y_proba = gbm.predict(X_test, num_iteration=best_iteration)
    y_pred = (y_proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"  AUC-ROC:   {auc_roc:.6f}")
    print(f"  Accuracy:  {accuracy:.6f}")
    print(f"  F1-Score:  {f1:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print()
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    metrics = {
        "auc_roc": auc_roc,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "train_time_seconds": train_time,
        "best_iteration": best_iteration,
    }

    return gbm, X_test, y_test, metrics


def benchmark_inference(model, X_test):
    """Measure inference latency and throughput"""
    print("=" * 60)
    print("STEP 4: Inference Benchmark")
    print("=" * 60)

    # Single row inference latency (average over 1000 runs)
    single_row = X_test.iloc[[0]]
    latencies = []
    for _ in range(1000):
        start = time.time()
        model.predict(single_row)
        latencies.append((time.time() - start) * 1000)  # ms

    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    print(f"  Single-row inference (1000 runs):")
    print(f"    Avg latency:  {avg_latency:.3f} ms")
    print(f"    P50 latency:  {p50_latency:.3f} ms")
    print(f"    P95 latency:  {p95_latency:.3f} ms")
    print(f"    P99 latency:  {p99_latency:.3f} ms")
    print()

    # Batch inference throughput (1000 rows)
    batch = X_test.iloc[:1000]
    start = time.time()
    model.predict(batch)
    batch_time = (time.time() - start) * 1000  # ms
    throughput = 1000 / (batch_time / 1000)  # rows/sec

    print(f"  Batch inference (1000 rows):")
    print(f"    Total time:  {batch_time:.3f} ms")
    print(f"    Throughput:  {throughput:.0f} rows/sec")
    print()

    return {
        "single_row_avg_latency_ms": round(avg_latency, 3),
        "single_row_p50_latency_ms": round(p50_latency, 3),
        "single_row_p95_latency_ms": round(p95_latency, 3),
        "single_row_p99_latency_ms": round(p99_latency, 3),
        "batch_1000_rows_time_ms": round(batch_time, 3),
        "batch_throughput_rows_per_sec": round(throughput, 0),
    }


def save_results(load_time, train_metrics, inference_metrics):
    """Save benchmark results to JSON"""
    print("=" * 60)
    print("STEP 5: Saving Results")
    print("=" * 60)

    results = {
        "benchmark_info": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dataset": "Credit Card Fraud Detection (mlg-ulb/creditcardfraud)",
            "dataset_rows": 284807,
            "model": "LightGBM (Gradient Boosting)",
            "instance_type": "CPU (m7i-flex.large — 2 vCPU, 8 GB RAM)",
            "lab": "Lab 16 - Cloud AI Environment Setup (CPU Fallback)",
        },
        "performance": {
            "data_load_time_seconds": round(load_time, 3),
            "training_time_seconds": round(train_metrics["train_time_seconds"], 3),
            "best_iteration": train_metrics["best_iteration"],
            "auc_roc": round(train_metrics["auc_roc"], 6),
            "accuracy": round(train_metrics["accuracy"], 6),
            "f1_score": round(train_metrics["f1_score"], 6),
            "precision": round(train_metrics["precision"], 6),
            "recall": round(train_metrics["recall"], 6),
        },
        "inference": inference_metrics,
    }

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {RESULT_FILE}")
    print()

    # Print summary table (for README 7.6)
    print("=" * 60)
    print("BENCHMARK RESULTS SUMMARY (Bảng 7.6)")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Kết quả':>20}")
    print(f"  {'-'*35} {'-'*20}")
    print(f"  {'Thời gian load data':<35} {load_time:>17.3f} s")
    print(f"  {'Thời gian training':<35} {train_metrics['train_time_seconds']:>17.3f} s")
    print(f"  {'Best iteration':<35} {train_metrics['best_iteration']:>20}")
    print(f"  {'AUC-ROC':<35} {train_metrics['auc_roc']:>20.6f}")
    print(f"  {'Accuracy':<35} {train_metrics['accuracy']:>20.6f}")
    print(f"  {'F1-Score':<35} {train_metrics['f1_score']:>20.6f}")
    print(f"  {'Precision':<35} {train_metrics['precision']:>20.6f}")
    print(f"  {'Recall':<35} {train_metrics['recall']:>20.6f}")
    print(f"  {'Inference latency (1 row)':<35} {inference_metrics['single_row_avg_latency_ms']:>16.3f} ms")
    print(f"  {'Inference throughput (1000 rows)':<35} {inference_metrics['batch_throughput_rows_per_sec']:>14.0f} rows/s")
    print("=" * 60)

    return results


def save_model(model):
    """Save trained model for the API server to load"""
    import pickle
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {MODEL_FILE}")
    print("  Restart ml-api service to load the model for live predictions.")
    print()


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  LightGBM Benchmark — Credit Card Fraud Detection       ║")
    print("║  Lab 16 CPU Fallback (Phần 7 README_aws.md)             ║")
    print("╚" + "═" * 58 + "╝")
    print()

    total_start = time.time()

    # Step 1: Load data
    df, load_time = load_data()

    # Step 2-3: Train and evaluate
    model, X_test, y_test, train_metrics = train_model(df)

    # Step 4: Inference benchmark
    inference_metrics = benchmark_inference(model, X_test)

    # Step 5: Save results
    results = save_results(load_time, train_metrics, inference_metrics)

    # Save model for API
    save_model(model)

    total_time = time.time() - total_start
    print(f"  Total benchmark time: {total_time:.3f}s")
    print()
    print("Done! Next steps:")
    print("  1. Copy benchmark_result.json for submission")
    print("  2. Take screenshot of this terminal output")
    print("  3. Check AWS Billing after 1 hour")
    print("  4. Run 'terraform destroy' when done!")


if __name__ == "__main__":
    main()
