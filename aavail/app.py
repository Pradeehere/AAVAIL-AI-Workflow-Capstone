from __future__ import annotations

import os
from flask import Flask, jsonify, request

from .logger import read_logs
from .model import train_all, predict


def create_app(config=None):
    app = Flask(__name__)
    app.config.update({
        "DATA_DIR": os.getenv("AAVAIL_DATA_DIR", "data/cs-train"),
        "MODEL_DIR": os.getenv("AAVAIL_MODEL_DIR", "models"),
        "TRAIN_LOG": os.getenv("AAVAIL_TRAIN_LOG", "logs/train.jsonl"),
        "PREDICT_LOG": os.getenv("AAVAIL_PREDICT_LOG", "logs/predict.jsonl"),
    })
    if config:
        app.config.update(config)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/train")
    def train_endpoint():
        payload = request.get_json(silent=True) or {}
        try:
            result = train_all(
                payload.get("data_dir", app.config["DATA_DIR"]),
                app.config["MODEL_DIR"],
                app.config["TRAIN_LOG"],
                fast=bool(payload.get("fast", False)),
            )
            return jsonify({"status": "trained", "models": result})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/predict")
    def predict_endpoint():
        payload = request.get_json(silent=True) or {}
        country = payload.get("country")
        target_date = payload.get("date")
        if not country or not target_date:
            return jsonify({"error": "country and date are required"}), 400
        try:
            result = predict(
                country, target_date,
                app.config["DATA_DIR"],
                app.config["MODEL_DIR"],
                app.config["PREDICT_LOG"],
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/logs")
    def logs_endpoint():
        kind = request.args.get("kind", "predict")
        path = app.config["TRAIN_LOG"] if kind == "train" else app.config["PREDICT_LOG"]
        return jsonify(read_logs(path))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
