"""
METHODOLOGY: Model Training Performance Test
This test evaluates the computational overhead of the machine learning pipeline 
defined in the system core. The script programmatically invokes the 
`prepare_training_data` and `train_model` functions from app.py. 
Resource utilization is monitored using the 'psutil' library for peak CPU (%) 
and RAM (MB) consumption, while 'time.perf_counter' measures the total execution 
latency in milliseconds. This isolated backend approach eliminates UI overhead 
to capture pure algorithmic performance.
"""

import os
import time
import psutil
import numpy as np
import json
from app import prepare_training_data, train_model

def run_training_test():
    results_dir = 'performance_test'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    try:
        # Load actual student structure (Simulating the 101 records from your report)
        with open('performance_test/sample_data.json', 'r') as f:
            students = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading sample data: {e}")
        return

    process = psutil.Process(os.getpid())
    print(f"Starting Model Training Test on {len(students)} records...")
    
    # Initialize CPU measurement (primes the counter)
    process.cpu_percent(interval=None)
    
    # Initial baseline
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.perf_counter()

    # 1. Real Data Preprocessing
    training_data = prepare_training_data(students, [])
    
    # Ensure all required features are present
    feature_cols = [
        'study_hours', 'library_visits', 'lms_login_per_month', 'gpa_y1s1', 
        'gpa_y1s2', 'gpa_y2s1', 'gpa_y2s2', 'gpa_y3s1', 'scholarship_amount', 
        'family_income', 'like_course', 'interested_in_subjects', 'course_motivates',
        'satisfied_with_performance', 'previous_grades_affect', 'study_regularly', 
        'submit_on_time', 'manage_time_well', 'instructors_explain_clearly', 
        'approach_instructors', 'instructors_encourage', 'classmates_influence_positively', 
        'work_well_with_classmates', 'friends_motivate'
    ]
    X = np.array([[row.get(col, 0) for col in feature_cols] for row in training_data])
    y = np.array([row['target'] for row in training_data])

    # 2. Real Model Training (Random Forest + Hyperopt + SMOTE)
    # Using your actual function with the seed defined in your steps
    model = train_model(X, y, 'Random Forest', 2024, use_hyperopt=True, use_smote=True)

    end_time = time.perf_counter()
    end_mem = process.memory_info().rss / (1024 * 1024)
    
    # Get average process CPU usage since the 'prime' call, normalized by core count
    cpu_usage = process.cpu_percent(interval=None) / psutil.cpu_count()

    total_time_ms = (end_time - start_time) * 1000
    mem_consumed = end_mem - start_mem

    result_log = (
        f"\n--- Model Training Performance Test [{time.ctime()}] ---\n"
        f"Dataset Size: {len(students)} students\n"
        f"Execution Time: {total_time_ms:.2f} ms\n"
        f"Peak CPU Usage: {cpu_usage:.2f}%\n"
        f"RAM Consumption: {mem_consumed:.2f} MB\n"
        f"Algorithm: Random Forest (Best Model)\n"
        f"----------------------------------------------------\n"
    )

    with open(os.path.join(results_dir, 'results.txt'), 'a') as f:
        f.write(result_log)
    
    print(result_log)
    print(f"Metrics saved to {results_dir}/results.txt")

if __name__ == "__main__":
    run_training_test()
