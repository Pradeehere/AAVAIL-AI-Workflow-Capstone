from aavail.data import fetch_data, convert_to_ts
from aavail.model import compare_models, train_all, predict
from .helpers import write_synthetic_aavail_json


def test_multiple_models_are_compared(tmp_path):
    data_dir = write_synthetic_aavail_json(tmp_path / "data", days=430)
    df = fetch_data(data_dir)
    ts = convert_to_ts(df)
    table, _, _ = compare_models(ts, fast=True)
    assert {"baseline_mean", "ridge", "random_forest", "gradient_boosting"}.issubset(set(table["model"]))
    assert table.iloc[0]["rmse"] >= 0


def test_train_and_predict_for_all_and_country(tmp_path):
    data_dir = write_synthetic_aavail_json(tmp_path / "data", days=430)
    model_dir = tmp_path / "models"
    train_log = tmp_path / "logs" / "train.jsonl"
    pred_log = tmp_path / "logs" / "predict.jsonl"
    result = train_all(data_dir, model_dir, train_log, fast=True)
    assert "all" in result
    assert "United Kingdom" in result

    p_all = predict("all", "2019-02-15", data_dir, model_dir, pred_log)
    p_uk = predict("United Kingdom", "2019-02-15", data_dir, model_dir, pred_log)
    assert p_all["prediction_30_day_revenue"] >= 0
    assert p_uk["prediction_30_day_revenue"] >= 0
