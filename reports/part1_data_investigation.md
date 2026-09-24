# Part 1 — Data investigation

## Business opportunity
AAVAIL management needs an accurate estimate of the next 30 days of revenue so managers and executives can make better planning decisions. In data-science terms this is a **supervised regression / time-series forecasting** problem because the output is numeric future revenue.

## Testable hypotheses
1. Recent revenue contains predictive signal for the next 30 days of revenue.
2. Revenue has seasonal structure, especially around late-year months.
3. Recent invoice activity and stream-view activity add signal beyond revenue history alone.
4. Country-level revenue behavior differs enough that country-specific models may outperform one global model for some markets.

## Ideal data
The ideal dataset contains timestamped transactions, country, invoice identifiers, revenue/price, stream IDs, view counts, customer identifiers, product metadata, pricing/promotions, acquisition channel, and calendar/event features. The provided data support the core revenue forecast but product/marketing context would improve causal interpretation.

## EDA
Run:

```bash
python scripts/download_data.py
python scripts/run_eda.py
```

The script creates:
- `reports/figures/top_countries_revenue.png`
- `reports/figures/daily_revenue.png`
- `reports/figures/monthly_seasonality.png`

The course dataset investigation should confirm the findings seen in the capstone material: the date range is roughly 500 days, the United Kingdom contributes the greatest total revenue, and activity is strongest around November–December. Re-run the script from the repository so the reported figures are generated from the raw JSON files rather than hard-coded values.
