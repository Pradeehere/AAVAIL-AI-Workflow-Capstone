from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def write_synthetic_aavail_json(root: Path, days=520):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2018-01-01")
    rows_by_month = {}
    countries = ["United Kingdom", "EIRE", "USA"]

    for offset in range(days):
        date = start + pd.Timedelta(days=offset)
        seasonal = 1.35 if date.month in (11, 12) else 1.0
        for country_idx, country in enumerate(countries):
            n = 2 + (offset + country_idx) % 3
            for j in range(n):
                price = (18 + 4 * country_idx + rng.normal(0, 2)) * seasonal
                row = {
                    "country": country,
                    "customer_id": int(1000 + offset * 10 + j + country_idx),
                    "day": date.day,
                    "invoice": f"A{offset:05d}{country_idx}{j}",
                    "month": date.month,
                    "price": float(max(1.0, price)),
                    "stream_id": int((offset * 3 + j) % 80),
                    "times_viewed": int(1 + (offset + j) % 6),
                    "year": date.year,
                }
                key = date.strftime("%Y-%m")
                rows_by_month.setdefault(key, []).append(row)

    # Exercise alias normalization on the first file.
    first = True
    for ym, rows in rows_by_month.items():
        if first:
            for r in rows:
                r["StreamID"] = r.pop("stream_id")
                r["TimesViewed"] = r.pop("times_viewed")
                r["total_price"] = r.pop("price")
            first = False
        (root / f"invoices-{ym}.json").write_text(json.dumps(rows), encoding="utf-8")
    return root
