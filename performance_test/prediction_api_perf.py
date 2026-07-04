"""
METHODOLOGY: Prediction Generation Performance Test (Real API Endpoint)
This script benchmarks the real backend prediction endpoint `POST /api/predict` by sending real student feature payloads.
It uses the `requests` library to measure round-trip latency in milliseconds.

It performs two programmatic scenarios:
1) A single prediction request for 1 student.
2) A batch prediction request for 100 students.

This is an isolated backend-level performance test: it does not involve UI rendering; it only calls the backend API.
"""

import os
import time
import json
import sys

# Ensure repository root is on sys.path so that backend imports/endpoints work consistently
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests


def _load_real_student_payloads(limit: int = 100):
    import pandas as pd
    
    data_dir = os.path.join(REPO_ROOT, "data")
    csv_files = ["student_performance_dataset_v2.csv", "student_performance_dataset.csv"]
    
    for f in csv_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Simplify payload creation by converting dataframe rows to dictionaries
            return df.head(limit).to_dict('records')
            
    raise FileNotFoundError(f"Could not find dataset in {data_dir}")


def request_single(url: str, student: dict):
    payload = {"students": [student]}
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, resp.json()


def request_batch_100(url: str, students_100: list[dict]):
    payload = {"students": students_100}
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    ms = (time.perf_counter() - t0) * 1000.0
    data = resp.json()
    return ms, data


def run_prediction_perf(base_url: str = "http://127.0.0.1:5000"):
    results_dir = os.path.join(os.getcwd(), "performance_test")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "results.txt")

    url = base_url + "/api/predict"

    # Ensure backend is reachable (isolated health check)
    health = requests.get(base_url + "/api/health", timeout=10)
    health.raise_for_status()

    students = _load_real_student_payloads(limit=100)
    if not students:
        raise RuntimeError("No student rows loaded from CSV to build prediction payloads.")

    # Single (1 student)
    single_ms_list = []
    for _ in range(3):
        ms, _ = request_single(url, students[0])
        single_ms_list.append(ms)

    # Batch (100 students)
    batch_students = students[:100]
    batch_ms_list = []
    for _ in range(3):
        ms, _ = request_batch_100(url, batch_students)
        batch_ms_list.append(ms)

    single_avg = sum(single_ms_list) / len(single_ms_list)
    batch_avg = sum(batch_ms_list) / len(batch_ms_list)

    result_log = (
        f"\n--- Prediction API Performance Test [{time.ctime()}] ---\n"
        f"Endpoint: POST /api/predict ({base_url})\n"
        f"Single Request Avg: {single_avg:.3f} ms (n=3)\n"
        f"Batch (100) Request Avg: {batch_avg:.3f} ms (n=3)\n"
        f"----------------------------------------------------\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(result_log)

    print(result_log)


if __name__ == "__main__":
    run_prediction_perf()
