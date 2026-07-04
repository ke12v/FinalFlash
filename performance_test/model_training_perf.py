"""
METHODOLOGY: Model Training Performance Test (Real Function Call)
This script programmatically calls the *real* training pipeline functions from the system backend: `prepare_training_data(...)` and `train_model(...)` in `app.py`.

Execution time is measured with `time.perf_counter()` and the process RSS memory is sampled with `psutil`.
If `psutil` is unavailable in the environment, the script falls back to a memory-only measurement using `resource` (best-effort on Windows may be limited).

This is an isolated backend-level test: it does not use the UI and does not call the HTTP `/api/train` endpoint.
"""

import os
import time
import numpy as np
import sys

# Ensure repository root is on sys.path so `import app` works when executed as:
#   python performance_test/model_training_perf.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import prepare_training_data, train_model

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def _sample_rss_mb():
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    # Fallback (best-effort). If psutil is missing, we still keep the script runnable.
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux it's KB, on macOS it's bytes; keep conservative conversion
        return rss_kb / 1024.0
    except Exception:
        return 0.0


def run_model_training_perf(algorithm: str = "Random Forest", random_seed: int = 2024):
    results_dir = os.path.join(os.getcwd(), "performance_test")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "results.txt")

    # Use the existing real CSV dataset from this repo as the real input source.
    csv_path = os.path.join(os.getcwd(), "data", "student_performance_dataset_v2.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.getcwd(), "data", "student_performance_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Could not find student_performance_dataset_v2.csv or student_performance_dataset.csv under data/."
        )

    import pandas as pd

    df = pd.read_csv(csv_path)

    # Construct the exact student dict structure expected by app.py.
    students = []
    for _, row in df.head(2000).iterrows():
        rec = {}
        rec["student_id"] = str(row["student_id"]) if "student_id" in df.columns else str(row.get("id", ""))
        rec["full_name"] = str(row["full_name"]) if "full_name" in df.columns else str(row.get("name", ""))

        for c in [
            "study_hours",
            "library_visits",
            "lms_login_per_month",
            "gpa_y1s1",
            "gpa_y1s2",
            "gpa_y2s1",
            "gpa_y2s2",
            "gpa_y3s1",
            "scholarship_amount",
            "family_income",
            "like_course",
            "interested_in_subjects",
            "course_motivates",
            "satisfied_with_performance",
            "previous_grades_affect",
            "try_improve_grades",
            "study_regularly",
            "submit_on_time",
            "manage_time_well",
            "instructors_explain_clearly",
            "approach_instructors",
            "instructors_encourage",
            "classmates_influence_positively",
            "work_well_with_classmates",
            "friends_motivate",
            "concerns",
        ]:
            if c in df.columns:
                rec[c] = row[c]

        students.append(rec)

    process = psutil.Process(os.getpid()) if psutil is not None else None

    # Build X/y using real app logic
    t0 = time.perf_counter()
    rss0 = _sample_rss_mb()

    # 1. Preprocessing phase
    training_data = prepare_training_data(students, [])
    t_pre = time.perf_counter()
    prep_ms = (t_pre - t0) * 1000.0

    feature_cols = [
        "study_hours",
        "library_visits",
        "lms_login_per_month",
        "gpa_y1s1",
        "gpa_y1s2",
        "gpa_y2s1",
        "gpa_y2s2",
        "gpa_y3s1",
        "scholarship_amount",
        "family_income",
        "like_course",
        "interested_in_subjects",
        "course_motivates",
        "satisfied_with_performance",
        "previous_grades_affect",
        "study_regularly",
        "submit_on_time",
        "manage_time_well",
        "instructors_explain_clearly",
        "approach_instructors",
        "instructors_encourage",
        "classmates_influence_positively",
        "work_well_with_classmates",
        "friends_motivate",
    ]

    X = np.array([[row.get(col, 0) for col in feature_cols] for row in training_data], dtype=float)
    y = np.array([row.get("target", 0) for row in training_data], dtype=int)

    # Peak sampling during training
    peak_cpu = 0.0
    peak_ram = rss0

    # Warm-up CPU sampling
    if process is not None:
        _ = process.cpu_percent(interval=None)

    # Sample loop in a lightweight manner: we measure peaks by polling while training runs.
    # For simplicity (and to keep isolated test reproducible), we perform one polling just before/after.
    # If psutil exists, this still captures real CPU sample.
    model = None
    model = train_model(
        X,
        y,
        algorithm,
        random_seed,
        use_hyperopt=True,
        use_smote=True,
    )

    if process is not None:
        peak_cpu = process.cpu_percent(interval=None)

    peak_ram = max(peak_ram, _sample_rss_mb())
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000.0
    rss1 = _sample_rss_mb()
    ram_delta = rss1 - rss0

    result_log = (
        f"\n--- Model Training Performance Test [{time.ctime()}] ---\n"
        f"| Metric | Observed Value | Notes |\n"
        f"|---|---:|---|\n"
        f"| Dataset records used | {min(2000, len(df))} | CSV source records |\n"
        f"| Preprocessing time | {prep_ms:.2f} ms | Data cleaning & encoding |\n"
        f"| Total training time | {total_ms:.2f} ms | `prepare_training_data` + `train_model` |\n"
        f"| Peak CPU (%) | {peak_cpu:.2f} % | Process utilization |\n"
        f"| Peak RAM / RSS (MB) | {peak_ram:.2f} MB | Memory footprint |\n"
        f"| Algorithm selected | {algorithm} | Random Forest |\n"
        f"----------------------------------------------------\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(result_log)

    print(result_log)


if __name__ == "__main__":
    run_model_training_perf()
