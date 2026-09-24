from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "country", "customer_id", "day", "invoice", "month",
    "price", "stream_id", "times_viewed", "year",
]

COLUMN_ALIASES = {
    "StreamID": "stream_id",
    "TimesViewed": "times_viewed",
    "total_price": "price",
}


def fetch_data(data_dir: str | Path) -> pd.DataFrame:
    """Load all AAVAIL invoice JSON files and return one cleaned DataFrame.

    The official capstone data contain small schema-name inconsistencies across
    months. This function normalizes those names, strips letters from invoice
    identifiers, creates a proper invoice_date, sorts chronologically, and
    fails fast on malformed input.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {data_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_json(path)
        frame = frame.rename(columns=COLUMN_ALIASES)
        missing = set(CANONICAL_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame = frame[CANONICAL_COLUMNS].copy()
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D+", "", regex=True)
    df["invoice_date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="raise",
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0)
    df = df.dropna(subset=["country", "invoice_date", "price"]).copy()
    df = df.sort_values("invoice_date").reset_index(drop=True)
    return df


def convert_to_ts(df: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Aggregate transaction rows into a complete daily time series."""
    work = df.copy()
    if country and country != "all":
        if country not in set(work["country"].astype(str)):
            raise ValueError(f"Country not found: {country}")
        work = work[work["country"] == country].copy()

    if work.empty:
        raise ValueError("No rows available after filtering")

    start = work["invoice_date"].min().normalize()
    stop = work["invoice_date"].max().normalize()
    full_dates = pd.date_range(start, stop, freq="D")

    grouped = work.groupby("invoice_date").agg(
        purchases=("invoice", "size"),
        unique_invoices=("invoice", "nunique"),
        unique_streams=("stream_id", "nunique"),
        total_views=("times_viewed", "sum"),
        revenue=("price", "sum"),
    )
    grouped = grouped.reindex(full_dates, fill_value=0)
    grouped.index.name = "date"
    grouped = grouped.reset_index()
    grouped["year_month"] = grouped["date"].dt.strftime("%Y-%m")
    return grouped


def top_countries_by_revenue(df: pd.DataFrame, n: int = 10) -> list[str]:
    totals = df.groupby("country")["price"].sum().sort_values(ascending=False)
    return totals.head(n).index.astype(str).tolist()


def make_time_series_map(df: pd.DataFrame, top_n: int = 10) -> dict[str, pd.DataFrame]:
    series = {"all": convert_to_ts(df)}
    for country in top_countries_by_revenue(df, top_n):
        series[country] = convert_to_ts(df, country)
    return series


def engineer_features(df: pd.DataFrame, training: bool = True):
    """Create supervised features for 30-day revenue forecasting.

    Features mirror the capstone guidance: prior revenue windows, same period
    one year earlier, recent invoice activity, and recent view activity.
    The target is total revenue over the next 30 days.
    """
    ts = df.copy().sort_values("date").reset_index(drop=True)
    ts["date"] = pd.to_datetime(ts["date"])
    dates = ts["date"].to_numpy(dtype="datetime64[D]")
    revenue = ts["revenue"].to_numpy(dtype=float)

    def between_sum(values: np.ndarray, start: np.datetime64, stop: np.datetime64) -> float:
        mask = (dates >= start) & (dates < stop)
        return float(values[mask].sum())

    features = defaultdict(list)
    y = np.zeros(len(ts), dtype=float)
    revenue_windows = [7, 14, 28, 70]

    invoices = ts["unique_invoices"].to_numpy(dtype=float)
    views = ts["total_views"].to_numpy(dtype=float)

    for i, day in enumerate(dates):
        current = np.datetime64(day, "D")
        for num in revenue_windows:
            features[f"previous_{num}"].append(
                between_sum(revenue, current - np.timedelta64(num, "D"), current)
            )

        plus_30 = current + np.timedelta64(30, "D")
        y[i] = between_sum(revenue, current, plus_30)

        prev_year_start = current - np.timedelta64(365, "D")
        prev_year_stop = plus_30 - np.timedelta64(365, "D")
        features["previous_year"].append(
            between_sum(revenue, prev_year_start, prev_year_stop)
        )

        recent_start = current - np.timedelta64(30, "D")
        recent_mask = (dates >= recent_start) & (dates < current)
        features["recent_invoices"].append(
            float(invoices[recent_mask].mean()) if recent_mask.any() else 0.0
        )
        features["recent_views"].append(
            float(views[recent_mask].mean()) if recent_mask.any() else 0.0
        )

    X = pd.DataFrame(features).fillna(0.0)
    useful = X.sum(axis=1).to_numpy() > 0
    X = X.loc[useful].reset_index(drop=True)
    y = y[useful]
    dates = dates[useful]

    if training and len(X) > 30:
        X = X.iloc[:-30].reset_index(drop=True)
        y = y[:-30]
        dates = dates[:-30]

    return X, y, dates
