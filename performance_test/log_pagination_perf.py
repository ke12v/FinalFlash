"""
METHODOLOGY: Log Viewing and Activity Monitoring (Real Log Pagination)
This script measures backend latency for paginated access to the real logs table `training_logs`.
It executes a SQL query with `LIMIT` and `OFFSET` to fetch exactly 100 records per page.

Execution time is measured using `time.perf_counter()` and the real PostgreSQL query is executed
via `psycopg2`.

This is an isolated backend-level performance test: it does not use any UI or network calls beyond DB.
"""

import os
import time
import sys

# Ensure repository root is on sys.path so that any backend-related imports behave consistently.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _get_conn():
    import psycopg2
    from psycopg2.extras import RealDictCursor

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url, cursor_factory=RealDictCursor)

    host = os.getenv("PGHOST", "localhost")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")
    dbname = os.getenv("PGDATABASE")
    port = int(os.getenv("PGPORT", "5432"))

    if not user or password is None or not dbname:
        raise RuntimeError(
            "Missing DB credentials. Provide DATABASE_URL or set PGHOST/PGUSER/PGPASSWORD/PGDATABASE."
        )

    return psycopg2.connect(
        host=host,
        user=user,
        password=password,
        dbname=dbname,
        port=port,
        cursor_factory=RealDictCursor,
    )


def run_log_pagination_perf(limit: int = 100, offset: int = 0):
    results_dir = os.path.join(os.getcwd(), "performance_test")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "results.txt")

    query = """
        SELECT
            id, algorithm, accuracy, precision, recall, f1_score, roc_auc,
            is_best, department, training_session_id, dataset_size, created_at
        FROM training_logs
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s;
    """

    start = time.perf_counter()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit, offset))
            rows = cur.fetchall()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result_log = (
        f"\n--- Log Pagination DB Query Performance Test [{time.ctime()}] ---\n"
        f"LIMIT: {limit} | OFFSET: {offset}\n"
        f"Pure execution time (ms): {elapsed_ms:.3f} ms\n"
        f"Records fetched: {len(rows)}\n"
        f"----------------------------------------------------\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(result_log)

    print(result_log)


if __name__ == "__main__":
    run_log_pagination_perf(limit=100, offset=0)

