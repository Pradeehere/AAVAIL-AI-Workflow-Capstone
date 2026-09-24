from aavail.logger import log_predict, log_train, read_logs


def test_train_and_predict_logs_are_isolated(tmp_path):
    train = tmp_path / "train.jsonl"
    pred = tmp_path / "predict.jsonl"
    log_train(train, "all", ("2018-01-01", "2019-01-01"), {"rmse": 2.0}, 0.2, "1.0")
    log_predict(pred, "all", "2019-01-01", 123.4, 0.01, "1.0")
    assert len(read_logs(train)) == 1
    assert read_logs(train)[0]["event"] == "train"
    assert len(read_logs(pred)) == 1
    assert read_logs(pred)[0]["event"] == "predict"
