# Part 2 — Model building and selection

## Forecast target
For each date, the target is **total revenue over the next 30 days**.

## Engineered features
- revenue over previous 7 days
- revenue over previous 14 days
- revenue over previous 28 days
- revenue over previous 70 days
- revenue for the corresponding 30-day period one year earlier
- average recent invoice count
- average recent stream views

## Models compared
1. Mean-value dummy regressor — baseline
2. Ridge regression
3. Random forest regressor
4. Gradient boosting regressor

Evaluation uses a chronological 80/20 split instead of randomly shuffling time-series observations. The selection metric is RMSE, with MAE and R² also reported. The best model is retrained on all available training examples and serialized with `joblib`.

Run:

```bash
python scripts/train.py
```

The code compares all candidate models for `all` and for each of the top revenue-producing countries. This satisfies the requirement to compare multiple models rather than committing to one approach before measurement.

## Baseline comparison visualization
`train_all()` writes `reports/figures/model_comparison_all.png`, a validation-RMSE bar chart comparing all candidate models, including the mean-value baseline. It also writes the underlying scores to `reports/figures/model_comparison_all.csv`.
