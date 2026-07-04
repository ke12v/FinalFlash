# Thesis Manuscript Sections: Isolated Backend Performance Tests

## 1) Model Training Performance Test
**Methodology.** This test evaluates the model training throughput of the system’s backend ML pipeline by invoking the *real* model-training logic directly from the codebase. Programmatically, the script imports and executes `prepare_training_data(...)` and `train_model(...)` from `app.py`, constructing the feature matrix and target labels using the project’s existing dataset schema (CSV under `data/`). Execution duration is captured using `time.perf_counter()` to measure end-to-end latency in milliseconds. For runtime resource profiling, the script samples the current process memory footprint and CPU usage using the `psutil` library (and includes a best-effort fallback if `psutil` is unavailable). The test is isolated at the backend level (no UI rendering and no HTTP round-trips), ensuring the measured metrics correspond to algorithmic and preprocessing cost only.

**Presentation of Data (Table Template).**

| Metric | Observed Value | Notes |
|---|---:|---|
| Dataset records used | [X] | e.g., first 2000 rows |
| Preprocessing time | [X] ms | if separately measured |
| Total training time | [X] ms | `prepare_training_data` + `train_model` |
| Peak CPU (%) | [Y] % | from `psutil` sampling |
| Peak RAM / RSS (MB) | [Y] MB | from `psutil` sampling |
| Algorithm selected | [Name] | e.g., Random Forest |

**Analysis and Interpretation of Results.** The measured total execution time ([X] ms) indicates the computational demand required to complete the training pipeline under realistic data-driven conditions, including data preprocessing, scaling, class balancing, and hyperparameter optimization as implemented in the backend. The peak CPU ([Y]%) reflects how intensively CPU resources are utilized during training, which is directly relevant to scalability and whether the system can meet training-time constraints in production. The peak RAM/RSS ([Y] MB) provides evidence of memory efficiency, signaling whether the training procedure can operate within typical server memory budgets without inducing swapping or instability. Together, these metrics constitute empirical evidence that the backend training workflow achieves measurable computational efficiency while remaining isolated from UI/network variability.

---

## 2) Prediction Generation Performance Test
**Methodology.** This test measures real prediction latency by calling the system’s actual backend prediction endpoint. The script uses the `requests` library to send HTTP `POST` requests to the locally running Flask route `POST /api/predict`. It constructs payloads using the project’s real dataset rows (CSV under `data/`) so that the features match what `app.py` expects during `prepare_training_data` for prediction. Two scenarios are tested: (1) a single prediction request for 1 student, and (2) a batch prediction request covering 100 students. Round-trip time is measured with `time.perf_counter()` and converted to milliseconds; the script computes average response time across repeated runs to reduce transient variance. Because UI is excluded and the test directly targets the backend API endpoint, measured performance reflects the isolated computational cost plus minimal backend request handling overhead.

**Presentation of Data (Table Template).**

| Scenario | Request Count | Batch Size | Avg Response Time | Latency Std Dev (optional) | Endpoint |
|---|---:|---:|---:|---:|---|
| Single Request | [n] | 1 | [X] ms | [X] ms | `POST /api/predict` |
| Batch Request | [n] | 100 | [Y] ms | [Y] ms | `POST /api/predict` |

**Analysis and Interpretation of Results.** The average single-request latency ([X] ms) quantifies the backend’s responsiveness for real-time student risk assessment. The batch request latency for 100 predictions ([Y] ms) demonstrates how performance scales when the system must score multiple records in one call. Comparing these metrics reveals whether the backend exhibits near-linear scaling or amortizes overhead effectively (e.g., model/scaler loading effects and request parsing). Efficient and predictable latency at both single and batch scales supports the thesis claim that the system can deliver timely predictions while maintaining acceptable computational performance, independent of UI rendering and external network delays.

---

## 3) Report Viewing and Metric Generation Performance Test
**Methodology.** This test measures the backend execution time required to generate report metrics by directly executing the real SQL logic used by the reporting layer on the system database. The script establishes a real PostgreSQL connection using `psycopg2` (or `DATABASE_URL`) and executes a production-relevant query against the `training_logs` table, aggregating metrics such as average accuracy, precision, recall, F1-score, and ROC AUC grouped by algorithm and department. The query is timed using `time.perf_counter()` to obtain pure execution latency in milliseconds, covering SQL execution and row fetch from the database cursor. No UI requests are performed; therefore, the metric captures backend database performance only, aligning with the thesis objective of isolating performance at the backend level.

**Presentation of Data (Table Template).**

| Metric | Observed Value | Notes |
|---|---:|---|
| Query execution time | [X] ms | `time.perf_counter()` |
| Rows returned | [X] | grouped results |
| Database table queried | `training_logs` | real system table |
| Aggregated metrics included | [X] | accuracy/F1/AUC/etc. |

**Analysis and Interpretation of Results.** The pure SQL execution time ([X] ms) serves as evidence of how efficiently the database can compute and return aggregated reporting metrics under realistic conditions. Because the query targets indexed fields and performs deterministic aggregation (e.g., `AVG` and `GROUP BY`), the resulting latency provides empirical support for the system’s report generation scalability as historical training logs grow. The number of rows returned ([X]) also helps interpret the workload imposed on the application layer, indicating whether reporting remains lightweight or becomes bottlenecked due to large result sets. Collectively, these findings validate system efficiency at the database-backend boundary and demonstrate that report metric generation is performant without requiring UI involvement.

---

## 4) Log Viewing and Activity Monitoring (Real Log Pagination) Performance Test
**Methodology.** This test assesses the performance of paginated backend log retrieval, which is critical for activity monitoring and auditability as the number of training runs increases. The script connects to the real PostgreSQL database using `psycopg2` and executes an actual paginated query over the `training_logs` table using SQL `LIMIT` and `OFFSET` semantics. It fetches exactly 100 log records per page and times the end-to-end database query execution with `time.perf_counter()`, reporting the execution time in milliseconds. By using real database access and measuring the query cost directly, this test isolates backend performance from UI rendering and avoids network variability; the measured latency therefore reflects how the system will behave when users navigate log pages.

**Presentation of Data (Table Template).**

| Pagination Parameters | Observed Execution Time | Records Fetched | Table |
|---|---:|---:|---|
| LIMIT = 100, OFFSET = [X] | [Y] ms | 100 | `training_logs` |

**Analysis and Interpretation of Results.** The measured pagination execution time ([Y] ms) quantifies the backend efficiency of retrieving a fixed-size page of audit logs using `LIMIT`/`OFFSET`. This metric is central to demonstrating responsiveness in administrative monitoring interfaces: stable and low execution time suggests the system can support frequent browsing even when logs accumulate. If performance remains consistent across different OFFSET values, it indicates that indexing and query design sufficiently control latency growth. If latency increases significantly for deeper offsets, the result highlights opportunities for optimization (e.g., keyset pagination). Regardless, the empirical results provide thesis-grade evidence about backend pagination performance under isolated conditions.

