from aavail.app import create_app
from aavail.model import train_all
from .helpers import write_synthetic_aavail_json


def test_health_and_prediction_api(tmp_path):
    data_dir = write_synthetic_aavail_json(tmp_path / "data", days=430)
    model_dir = tmp_path / "models"
    train_all(data_dir, model_dir, tmp_path / "train.jsonl", fast=True)

    app = create_app({
        "TESTING": True,
        "DATA_DIR": str(data_dir),
        "MODEL_DIR": str(model_dir),
        "TRAIN_LOG": str(tmp_path / "api-train.jsonl"),
        "PREDICT_LOG": str(tmp_path / "api-predict.jsonl"),
    })
    client = app.test_client()

    assert client.get("/health").get_json()["status"] == "ok"
    bad = client.post("/predict", json={"country": "all"})
    assert bad.status_code == 400

    ok = client.post("/predict", json={"country": "all", "date": "2019-02-15"})
    assert ok.status_code == 200
    assert "prediction_30_day_revenue" in ok.get_json()
