"""
METHODOLOGY: Report Viewing and Metric Generation (Real Database Query)
This script measures backend report generation latency by executing a real SQL query against
`training_logs` (the same table used for report/metric views in the system).

It uses `psycopg2` for a real PostgreSQL connection and `time.perf_counter()` to measure pure execution time.
No UI calls are made—only direct DB query execution.

Connection is configured via environment variables:
- DATABASE_URL (preferred)
- or PGHOST/PGUSER/PGPASSWORD/PGDATABASE/PGPORT
"""

import os
import time


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


def run_report_query_perf():
    results_dir = os.path.join(os.getcwd(), "performance_test")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "results.txt")

    # Example "report" query: top algorithms by average weighted F1 across departments.
    query = """
        SELECT
            department,
            algorithm,
            COUNT(*) AS runs,
            AVG(f1_score)::float AS avg_f1_score,
            AVG(accuracy)::float AS avg_accuracy,
            AVG(roc_auc)::float AS avg_roc_auc
        FROM training_logs
        WHERE f1_score IS NOT NULL
        GROUP BY department, algorithm
        ORDER BY runs DESC
        LIMIT 50;
    """

    start = time.perf_counter()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result_log = (
        f"\n--- Report DB Query Performance Test [{time.ctime()}] ---\n"
        f"Pure execution time (ms): {elapsed_ms:.3f} ms\n"
        f"Rows returned: {len(rows)}\n"
        f"----------------------------------------------------\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(result_log)

    print(result_log)


if __name__ == "__main__":
    run_report_query_perf()

