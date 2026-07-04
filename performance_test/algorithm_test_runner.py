import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Ensure repository root is on sys.path to import app.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import prepare_training_data, train_model

def run_algorithm_performance_test():
    print("--- Starting Algorithm Performance Test ---")
    
    # STEP 1: Environment Setup
    print("[Step 1/8] Setting up environment and output directories...")
    # Redirecting visualization output to the reports folder as requested
    results_dir = os.path.join(REPO_ROOT, "reports")
    os.makedirs(results_dir, exist_ok=True)
    
    # STEP 2: Load Real Dataset
    print("[Step 2/8] Loading student performance dataset (v2)...")
    csv_path = os.path.join(REPO_ROOT, "data", "student_performance_dataset_v2.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Convert to student dict structure expected by system
    students = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["student_id"] = str(row.get("student_id", row.get("id", "")))
        students.append(rec)

    # 3. Use Actual Backend Preprocessing
    print("Preprocessing actual dataset...")
    training_data = prepare_training_data(students, [])
    
    # Feature set derived from your actual database schema
    feature_cols = [
        "study_hours", "library_visits", "lms_login_per_month", "gpa_y1s1", "gpa_y1s2",
        "gpa_y2s1", "gpa_y2s2", "gpa_y3s1", "scholarship_amount", "family_income",
        "like_course", "interested_in_subjects", "course_motivates", "satisfied_with_performance",
        "previous_grades_affect", "try_improve_grades", "study_regularly", 
        "submit_on_time", "manage_time_well",
        "instructors_explain_clearly", "approach_instructors", "instructors_encourage",
        "classmates_influence_positively", "work_well_with_classmates", "friends_motivate"
    ]

    X = np.array([[row.get(col, 0) for col in feature_cols] for row in training_data], dtype=float)
    y = np.array([row.get("target", 0) for row in training_data], dtype=int)
    y = np.where(y >= 1, 1, 0) # Force binary: 1 and 2 become 1

    # STEP 5: Dataset Splitting
    print("[Step 5/8] Performing stratified 70/30 train-test split for valid evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )

    # STEP 6: Model Training & Optimization
    print("[Step 6/8] Training Random Forest model with SMOTE balancing and Hyperopt tuning...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_model(X_train_scaled, y_train, "Random Forest", 123, use_hyperopt=True, use_smote=True)

    # STEP 7: Evaluation
    print("[Step 7/8] Generating predictions and calculating classification metrics...")
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-Score": f1_score(y_test, y_pred, average='weighted'),
        "CV Accuracy Mean": 0.0,
        "CV Std. Dev.": 0.0
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    metrics["CV Accuracy Mean"] = cv_scores.mean()
    metrics["CV Std. Dev."] = cv_scores.std()

    # STEP 8: Result Export & Visualization
    print("[Step 8/8] Exporting results and generating confusion matrix plot...")
    res_path = os.path.join(results_dir, "algorithm_results.txt")
    
    methodology_text = (
        "METHODOLOGY:\n"
        "1. Data Acquisition: Loaded 101 records from student_performance_dataset_v2.csv.\n"
        "2. Preprocessing: Mapped survey and GPA data using the production 'prepare_training_data' pipeline.\n"
        "3. Class Balancing: Applied SMOTE to training subsets to handle minority class representation.\n"
        "4. Optimization: Utilized GridSearchCV/Hyperopt for parameter selection (Random Forest).\n"
        "5. Validation: Stratified Train/Test split (70/30) used for final metric computation.\n"
    )

    with open(res_path, "w", encoding="utf-8") as f:
        f.write(f"--- Algorithm Performance Test Results ---\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"\n{methodology_text}\n")
        f.write(f"Model: Random Forest (Optimized)\n")
        f.write(f"Test Records: {len(y_test)}\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Good Standing', 'At-Risk']))

    # 8. Generate and Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good Standing', 'At-Risk'],
                yticklabels=['Good Standing', 'At-Risk'])
    plt.title('Confusion Matrix: Random Forest')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nTest Complete.")
    print(f"Metrics saved to: {res_path}")
    print(f"Confusion Matrix saved to: {cm_path}")
    print("-" * 30)
    for m, v in metrics.items():
        print(f"{m}: {v*100:.2f}%")

if __name__ == "__main__":
    run_algorithm_performance_test()
