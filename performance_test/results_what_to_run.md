# What to run (backend-only performance tests)

All scripts below append results to:
- `performance_test/results.txt`

## 0) Ensure backend is running (for prediction API test)
Start Flask server in one terminal:
- `python app.py`

Keep it running.

## 1) Model Training Performance Test (real function call)
- `python performance_test/model_training_perf.py`

## 2) Prediction API Performance Test (real endpoint)
- `python performance_test/prediction_api_perf.py`

## 3) Report Viewing & Metric Generation (real DB query)
- `python performance_test/report_generation_db_perf.py`

Requires DB credentials via env vars:
- `DATABASE_URL` (preferred)
  OR
- `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, `PGPORT`

## 4) Log Viewing & Activity Monitoring (real DB pagination)
- `python performance_test/log_pagination_perf.py`

Requires same DB credentials as (3).

