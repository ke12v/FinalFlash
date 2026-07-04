"""
METHODOLOGY: Prediction Generation Performance Test
This test benchmarks the responsiveness of the Prediction API endpoint. 
Using the 'requests' library, the script executes automated HTTP POST requests 
to the local '/api/predict' route. Two critical scenarios are measured: 
Individual Prediction (1 record) and Batch Prediction (100 records). 
By measuring the round-trip time in milliseconds (ms), the test establishes 
the system's capability to provide real-time risk assessment feedback.
"""

import requests
import time
import os
import json

def run_prediction_perf():
    results_dir = 'performance_test'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    url = "http://127.0.0.1:5000/api/predict"
    
    # Load real student data structure
    with open('performance_test/sample_data.json', 'r') as f:
        full_data = json.load(f)
    
    single_payload = {"students": [full_data[0]]}
    batch_payload = {"students": full_data[:100]} if len(full_data) >= 100 else {"students": full_data}

    print("Benchmarking API Latency...")

    # Test Single
    start = time.perf_counter()
    requests.post(url, json=single_payload)
    single_ms = (time.perf_counter() - start) * 1000

    # Test Batch
    start = time.perf_counter()
    requests.post(url, json=batch_payload)
    batch_ms = (time.perf_counter() - start) * 1000

    result_log = (
        f"\n--- Prediction API Latency Test [{time.ctime()}] ---\n"
        f"Single Request Latency: {single_ms:.2f} ms\n"
        f"Batch (100) Request Latency: {batch_ms:.2f} ms\n"
        f"Throughput: {len(batch_payload['students'])/ (batch_ms/1000):.2f} req/sec\n"
        f"----------------------------------------------------\n"
    )

    with open(os.path.join(results_dir, 'results.txt'), 'a') as f:
        f.write(result_log)
    
    print(result_log)

if __name__ == "__main__":
    # Ensure the Flask app is running before executing this
    try:
        requests.get("http://127.0.0.1:5000/api/health")
        run_prediction_perf()
    except:
        print("ERROR: System API (app.py) is not running on localhost:5000.")
        print("Please run 'python app.py' in another terminal first.")
