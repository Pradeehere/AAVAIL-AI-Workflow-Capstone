import pandas as pd
from aavail.data import fetch_data, convert_to_ts, engineer_features
from .helpers import write_synthetic_aavail_json


def test_ingestion_normalizes_and_sorts(tmp_path):
    data_dir = write_synthetic_aavail_json(tmp_path / "data", days=80)
    df = fetch_data(data_dir)
    assert {"stream_id", "times_viewed", "price", "invoice_date"}.issubset(df.columns)
    assert df["invoice_date"].is_monotonic_increasing
    assert df["invoice"].str.fullmatch(r"\d+").all()


def test_time_series_and_features(tmp_path):
    data_dir = write_synthetic_aavail_json(tmp_path / "data", days=430)
    df = fetch_data(data_dir)
    ts = convert_to_ts(df, "United Kingdom")
    X, y, dates = engineer_features(ts)
    assert len(ts) == 430
    assert len(X) == len(y) == len(dates)
    assert {"previous_7", "previous_28", "previous_year", "recent_views"}.issubset(X.columns)
