# Part 3 — Production, API, tests, monitoring

## API
The Flask application provides:
- `GET /health`
- `POST /train`
- `POST /predict` — requires `country` and `date`
- `GET /logs?kind=predict|train`

Example:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"all","date":"2019-07-01"}'
```

## Testing
`./run_tests.sh` runs all tests in one command. Separate test modules cover ingestion/feature engineering, model training/prediction, API behavior, and logging. Tests write to pytest temporary directories, so production models and production logs are not modified.

## Monitoring
`aavail.monitor.evaluate_production()` combines training and newly arrived production transactions, evaluates only production dates with a fully known 30-day future target, and reports RMSE/MAE. It also generates a time-series visualization of predicted versus known values. This is the primary model-performance feedback loop.

## Containerization
`Dockerfile` installs dependencies, runs the complete test suite during build, and serves Flask through Gunicorn. `docker-compose.yml` mounts data, model, and log directories so state persists outside the container.
