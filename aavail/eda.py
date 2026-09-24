from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from .data import fetch_data, convert_to_ts


def run_eda(data_dir, output_dir="reports/figures"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = fetch_data(data_dir)

    revenue = df.groupby("country")["price"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    revenue.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top countries by total revenue")
    ax.set_xlabel("Revenue")
    fig.tight_layout()
    fig.savefig(out / "top_countries_revenue.png", dpi=160)
    plt.close(fig)

    ts = convert_to_ts(df)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts["date"], ts["revenue"], linewidth=1)
    ax.set_title("Daily revenue over time")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    fig.savefig(out / "daily_revenue.png", dpi=160)
    plt.close(fig)

    monthly = ts.assign(month=ts["date"].dt.month).groupby("month")["revenue"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_xticks(range(1, 13))
    ax.set_title("Average daily revenue by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average daily revenue")
    fig.tight_layout()
    fig.savefig(out / "monthly_seasonality.png", dpi=160)
    plt.close(fig)

    return {
        "rows": int(len(df)),
        "date_min": str(df["invoice_date"].min().date()),
        "date_max": str(df["invoice_date"].max().date()),
        "unique_invoice_dates": int(df["invoice_date"].nunique()),
        "top_country_by_revenue": str(revenue.index[0]),
        "figures": [str(p) for p in sorted(out.glob("*.png"))],
    }
