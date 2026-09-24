from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import fetch_data, make_time_series_map, engineer_features
from .model import _load_bundle


def evaluate_production(
    country,
    training_data_dir,
    production_data_dir,
    model_dir="models",
    output_dir="reports/figures",
):
    """Evaluate a trained model on dates that became available in production.

    Training and production transactions are combined before feature
    engineering so production-day features have their full historical context.
    Only dates from the production period with a known 30-day future target are
    scored.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = fetch_data(training_data_dir)
    prod_df = fetch_data(production_data_dir)
    prod_start = prod_df["invoice_date"].min()
    prod_stop = prod_df["invoice_date"].max()

    full_df = pd.concat([train_df, prod_df], ignore_index=True)
    full_df = full_df.drop_duplicates().sort_values("invoice_date").reset_index(drop=True)
    ts_map = make_time_series_map(full_df)
    if country not in ts_map:
        raise ValueError(f"Country not present in combined data: {country}")

    X, y, dates = engineer_features(ts_map[country], training=False)
    date_ts = pd.to_datetime(dates)
    # Need 30 known days after the target date to calculate the gold-standard target.
    valid_stop = prod_stop - pd.Timedelta(days=29)
    mask = (date_ts >= prod_start) & (date_ts <= valid_stop)
    if not mask.any():
        raise ValueError("Production period is too short to calculate a known 30-day target")

    X_eval = X.loc[mask].reset_index(drop=True)
    y_eval = y[mask]
    dates_eval = dates[mask]

    bundle = _load_bundle(country, model_dir)
    pred = bundle["model"].predict(X_eval[bundle["feature_columns"]])

    rmse = float(np.sqrt(mean_squared_error(y_eval, pred)))
    mae = float(mean_absolute_error(y_eval, pred))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(pd.to_datetime(dates_eval), y_eval, label="Known 30-day revenue")
    ax.plot(pd.to_datetime(dates_eval), pred, label="Predicted 30-day revenue")
    ax.set_title(f"Production performance — {country}")
    ax.set_ylabel("30-day revenue")
    ax.legend()
    fig.tight_layout()
    path = out / f"production_monitor_{str(country).replace(' ', '_')}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return {
        "country": country,
        "evaluation_start": str(pd.to_datetime(dates_eval[0]).date()),
        "evaluation_end": str(pd.to_datetime(dates_eval[-1]).date()),
        "rmse": rmse,
        "mae": mae,
        "figure": str(path),
    }
