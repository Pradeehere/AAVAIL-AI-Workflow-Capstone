from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .data import fetch_data, make_time_series_map, engineer_features
from .logger import log_predict, log_train

MODEL_VERSION = "1.0"
MODEL_NOTE = "30-day revenue forecast with time-aware validation"


def slug(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", country.lower()).strip("_")


def _metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def candidate_models(random_state=42, fast=False):
    trees = 40 if fast else 180
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=trees, max_depth=8, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=60 if fast else 140,
            max_depth=2,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }


def compare_models(ts_df: pd.DataFrame, fast=False):
    X, y, dates = engineer_features(ts_df, training=True)
    if len(X) < 80:
        raise ValueError("Not enough engineered rows to compare models")

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    rows = []
    fitted = {}
    for name, model in candidate_models(fast=fast).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = _metrics(y_test, pred)
        rows.append({"model": name, **score})
        fitted[name] = model

    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return table, fitted, (X, y, dates)


def train_all(data_dir, model_dir="models", log_path="logs/train.jsonl", fast=False, report_dir="reports/figures"):
    start = time.time()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_data(data_dir)
    ts_map = make_time_series_map(df)

    summary = {}
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for country, ts_df in ts_map.items():
        table, _, (X, y, dates) = compare_models(ts_df, fast=fast)
        if country == "all":
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ordered = table.sort_values("rmse", ascending=False)
            ax.barh(ordered["model"], ordered["rmse"])
            ax.set_title("Model comparison against baseline")
            ax.set_xlabel("Validation RMSE (lower is better)")
            fig.tight_layout()
            fig.savefig(report_dir / "model_comparison_all.png", dpi=160)
            plt.close(fig)
            table.to_csv(report_dir / "model_comparison_all.csv", index=False)
        best_name = table.iloc[0]["model"]
        best_model = candidate_models(fast=fast)[best_name]
        best_model.fit(X, y)

        bundle = {
            "country": country,
            "model_name": best_name,
            "model": best_model,
            "feature_columns": X.columns.tolist(),
            "version": MODEL_VERSION,
            "trained_through": str(pd.to_datetime(dates[-1]).date()),
        }
        out = model_dir / f"model-{slug(country)}.joblib"
        joblib.dump(bundle, out)

        metrics = table.iloc[0].drop("model").to_dict()
        log_train(
            log_path, country,
            (str(pd.to_datetime(dates[0]).date()), str(pd.to_datetime(dates[-1]).date())),
            metrics, time.time() - start, MODEL_VERSION, MODEL_NOTE,
        )
        summary[country] = {
            "best_model": best_name,
            "metrics": metrics,
            "artifact": str(out),
        }

    return summary


def _load_bundle(country, model_dir="models"):
    path = Path(model_dir) / f"model-{slug(country)}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model for country '{country}'. Run /train first.")
    return joblib.load(path)


def predict(country, target_date, data_dir, model_dir="models", log_path="logs/predict.jsonl"):
    start = time.time()
    target = np.datetime64(pd.to_datetime(target_date).date(), "D")
    bundle = _load_bundle(country, model_dir)

    df = fetch_data(data_dir)
    ts_map = make_time_series_map(df)
    if country not in ts_map:
        raise ValueError(f"Country not available: {country}")

    X, _, dates = engineer_features(ts_map[country], training=False)
    idx = np.where(dates == target)[0]
    if len(idx) == 0:
        lo = str(pd.to_datetime(dates[0]).date())
        hi = str(pd.to_datetime(dates[-1]).date())
        raise ValueError(f"Date {target_date} is outside the usable range {lo} to {hi}")

    row = X.iloc[[int(idx[0])]][bundle["feature_columns"]]
    value = float(bundle["model"].predict(row)[0])
    log_predict(log_path, country, target_date, value, time.time() - start, MODEL_VERSION)
    return {
        "country": country,
        "target_date": str(pd.to_datetime(target_date).date()),
        "prediction_30_day_revenue": round(value, 2),
        "model": bundle["model_name"],
        "model_version": bundle["version"],
    }
