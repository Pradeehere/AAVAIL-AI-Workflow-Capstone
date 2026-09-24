# AAVAIL 30-Day Revenue Forecast — IBM AI Enterprise Workflow Capstone

End-to-end capstone solution for the AAVAIL business opportunity: **predict total revenue over the next 30 days for a requested country and date**.

This repository covers all three capstone parts: automated ingestion/EDA, model comparison and training, Flask deployment, unit tests, Docker packaging, logging, and post-production monitoring.

## Project structure

```text
aavail/                 Python package
  data.py               ingestion, aggregation, feature engineering
  eda.py                EDA visualizations
  model.py              model comparison, training, prediction
  logger.py             isolated train/predict JSONL logs
  monitor.py            post-production performance monitoring
  app.py                Flask API
scripts/                 download/train/EDA/monitor entry points
tests/                   API, model, data, logging tests
reports/                 Part 1/2/3 written deliverables
Dockerfile               containerized API + test gate
docker-compose.yml       persistent data/models/logs
run_tests.sh             one-command test suite
```

## 1. Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download official capstone data

```bash
python scripts/download_data.py
```

This downloads the public files from `aavail/ai-workflow-capstone` into `data/cs-train` and `data/cs-production`.

## 3. Part 1 — investigate the data

```bash
python scripts/run_eda.py
```

See `reports/part1_data_investigation.md` and `reports/figures/`.

## 4. Part 2 — compare and train models

```bash
python scripts/train.py
```

The training code compares a baseline mean regressor, Ridge regression, Random Forest, and Gradient Boosting using a chronological holdout. The best RMSE model is retrained on all training data. Models are built for `all` plus the top ten countries by revenue. Training also writes `reports/figures/model_comparison_all.png`, which visually compares each model with the baseline.

## 5. Run every unit test

```bash
./run_tests.sh
```

Tests are isolated from production artifacts by using temporary directories.

## 6. Part 3 — run the API

```bash
python -m aavail.app
```

Train through the API:

```bash
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

Predict for all countries combined:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"all","date":"2019-07-01"}'
```

Predict for one country:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"United Kingdom","date":"2019-07-01"}'
```

View prediction logs:

```bash
curl "http://localhost:8080/logs?kind=predict"
```

## 7. Docker

```bash
docker compose build
docker compose up
```

The Docker build runs the full unit-test suite before creating the runnable image.

## 8. Production monitoring

After training, place/keep the official production months in `data/cs-production` and run:

```bash
python scripts/monitor.py
```

This computes production RMSE/MAE and writes a predicted-vs-known revenue visualization into `reports/figures/`.

## Peer-review rubric coverage

- API unit tests: `tests/test_api.py`
- Model unit tests: `tests/test_model.py`
- Logging unit tests: `tests/test_logger.py`
- All tests in one command: `run_tests.sh`
- Performance monitoring: `aavail/monitor.py`
- Test isolation from production logs/models: pytest `tmp_path`
- Predictions for a country and `all`: API + model tests
- Automated ingestion function/script: `aavail/data.py`, `scripts/download_data.py`
- Multiple models compared: four regressors in `aavail/model.py`
- EDA visualizations: `aavail/eda.py`
- Containerization: `Dockerfile`, `docker-compose.yml`
- Predicted-vs-known visualization: `aavail/monitor.py`

## Source data
Public capstone data and official guidance are from the AAVAIL repository:
`https://github.com/aavail/ai-workflow-capstone`
