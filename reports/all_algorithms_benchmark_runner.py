import pandas as pd
import numpy as np
import time
import psutil
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)

def measure_performance():
    # 1. Load Actual Dataset
    df = pd.read_csv('data/student_performance_dataset_v2.csv')
    
    # 2. Preprocessing
    # Drop identifiers and non-numeric columns
    cols_to_drop = ['student_id', 'name', 'course', 'department', 'email', 'scholarship', 'status']
    X = df.drop(columns=cols_to_drop + ['risk_label'])
    y = df['risk_label']
    
    # 2.1 Convert to Binary Classification (Mapping 1 and 2 to 'At-Risk')
    y = y.map({0: 0, 1: 1, 2: 1})
    
    # Fill missing values (for incomplete records in v2)
    X = X.fillna(X.mean())
    
    # Train-test split (70/30 Stratified as per system specs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE (Applied only to training data)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    algorithms = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    
    results = []
    process = psutil.Process(os.getpid())
    
    for name, clf in algorithms.items():
        # Computational Benchmarking Start
        mem_before = process.memory_info().rss / (1024 * 1024)
        cpu_start = psutil.cpu_percent(interval=None)
        
        # Training Time
        start_train = time.perf_counter()
        clf.fit(X_train_res, y_train_res)
        end_train = time.perf_counter()
        train_time_ms = (end_train - start_train) * 1000
        
        # Prediction Latency
        start_pred = time.perf_counter()
        y_pred = clf.predict(X_test_scaled)
        end_pred = time.perf_counter()
        pred_latency_ms = (end_pred - start_pred) * 1000
        
        # Computational Benchmarking End
        mem_after = process.memory_info().rss / (1024 * 1024)
        cpu_end = psutil.cpu_percent(interval=None)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            "Algorithm": name,
            "Records": len(df),
            "Features": X.shape[1],
            "TrainingExecutionTime_ms": round(train_time_ms, 4),
            "PredictionLatency_ms": round(pred_latency_ms, 4),
            "PeakMemoryUsage_MB": round(mem_after - mem_before, 4) if mem_after > mem_before else 0.01,
            "CPUUtilization_Percent": round(abs(cpu_end - cpu_start), 2),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_score": round(f1, 4)
        })
        
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/all_algorithms_benchmark_results.csv', index=False)
    
    # Save Summary
    with open('reports/all_algorithms_benchmark_summary.txt', 'w') as f:
        f.write("ALGORITHM PERFORMANCE BENCHMARK SUMMARY\n")
        f.write("========================================\n")
        f.write(f"Dataset: student_performance_dataset_v2.csv\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Train/Test Split: 70/30 (Stratified)\n")
        f.write(f"Preprocessing: Scaling, SMOTE (Training set)\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nConfusion Matrices can be generated via the runner script visualizer.\n")

if __name__ == "__main__":
    measure_performance()