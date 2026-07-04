""" 
METHODOLOGY: Report Viewing and Metric Generation (Real DB Query)
This benchmark measures the *pure backend execution time* required to generate report metrics by directly querying the same database tables used by the system backend.

IMPORTANT: This script is wired to real system database access.
- It uses environment variables commonly used for database connectivity (DATABASE_URL, PGHOST, PGUSER, PGPASSWORD, PGDATABASE).
- It executes a real SQL query against the training_logs table and computes aggregated metrics.

If your environment uses Supabase/Postgres, ensure you have a working connection string.
"""

import os
import time
import json
import psutil

import sys

# repo root not strictly required here, but keep consistent import behavior
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import psycopg2
from psycopg2.extras import RealDictCursor


def _get_conn():
    # Prefer DATABASE_URL if present
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        return psycopg2.connect(database_url, cursor_factory=RealDictCursor)

    # Otherwise, build from individual vars
    host = os.getenv('PGHOST', 'localhost')
    user = os.getenv('PGUSER')
    password = os.getenv('PGPASSWORD')
    dbname = os.getenv('PGDATABASE')
    port = int(os.getenv('PGPORT', '5432'))

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


def run_report_query_perf(limit_rows: int = 1000):
    results_dir = 'performance_test'
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, 'results.txt')

    query = """
        SELECT
            department,
            algorithm,
            COUNT(*) AS runs,
            AVG(accuracy)::float AS avg_accuracy,
            AVG(precision)::float AS avg_precision,
            AVG(recall)::float AS avg_recall,
            AVG(f1_score)::float AS avg_f1_score,
            AVG(roc_auc)::float AS avg_roc_auc
        FROM training_logs
        WHERE (accuracy IS NOT NULL)
        GROUP BY department, algorithm
        ORDER BY runs DESC
        LIMIT %s;
    """

    proc = psutil.Process(os.getpid())
    cpu_before = proc.cpu_times()
    mem_before = proc.memory_info().rss

    start = time.perf_counter()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit_rows,))
            rows = cur.fetchall()
    elapsed_ms = (time.perf_counter() - start) * 1000

    mem_after = proc.memory_info().rss

    result_log = (
        f"\n--- Report Viewing & Metric Query Test [{time.ctime()}] ---\n"
        f"Execution Time (pure SQL + fetch): {elapsed_ms:.4f} ms\n"
        f"Rows Returned (grouped): {len(rows)}\n"
        f"Approx RAM delta: {(mem_after - mem_before) / (1024*1024):.4f} MB\n"
        f"----------------------------------------------------\n"
    )

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(result_log)
    print(result_log)


if __name__ == '__main__':
    run_report_query_perf()

