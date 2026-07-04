Edit summary:
- The performance scripts import `from app import ...` / `import app`.
- When executed as `python performance_test/model_training_perf.py`, Python sets `sys.path[0]` to `performance_test/`, so `app.py` at repo root isn’t importable.

Fix implemented:
- For each performance script, prepend the repo root to `sys.path` before importing `app`.
- This keeps using the real functions/endpoints/db.

