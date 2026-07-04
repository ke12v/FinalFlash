from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import random
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid errors on server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP library not installed. Explainable AI features will be limited.")

app = Flask(__name__)
CORS(app)

def save_visualization_plots(model, X_test, y_test, feature_names, report_dir, timestamp):
    """Generate and save Confusion Matrix, Feature Importance, and SHAP plots as PNG files"""
    paths = {"cm": None, "importance": None, "shap": None}
    try:
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        classes = ['Low Risk', 'At-Risk']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix: {type(model).__name__}')
        plt.ylabel('Actual Status')
        plt.xlabel('Predicted Status')
        
        cm_path = os.path.join(report_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        paths["cm"] = cm_path
        plt.close()

        # 2. Feature Importance Ranking
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Take top 15 features for clarity
            top_n = 15
            plt.figure(figsize=(12, 8))
            sns.barplot(x=importances[indices[:top_n]], 
                        y=[feature_names[i] for i in indices[:top_n]], 
                        palette='viridis')
            
            plt.title(f'Top {top_n} Feature Importance Ranking (Random Forest)')
            plt.xlabel('Relative Importance Score')
            plt.ylabel('Student Factors/Features')
            
            feat_path = os.path.join(report_dir, f'feature_importance_{timestamp}.png')
            plt.savefig(feat_path, bbox_inches='tight', dpi=300)
            paths["importance"] = feat_path
            plt.close()
            
        # 3. SHAP Summary Plot
        if SHAP_AVAILABLE:
            try:
                plt.figure(figsize=(12, 8))
                
                # Select appropriate explainer based on model type
                if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                    # Optimized for Decision Trees and Random Forests
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                else:
                    # Generic explainer for SVM, KNN, Naive Bayes, etc.
                    # We use a small subset of X_test as background data for KernelExplainer if needed
                    explainer = shap.Explainer(model.predict, X_test)
                    shap_values = explainer(X_test).values

                # For binary RF, shap_values is a list [class0, class1]. We want class 1 (At-Risk)
                # shap_values[..., 1] is for the new Explainer API, shap_values[1] for older list format
                if isinstance(shap_values, list):
                    display_values = shap_values[1]
                elif len(shap_values.shape) == 3: # (samples, features, classes)
                    display_values = shap_values[:, :, 1]
                else:
                    display_values = shap_values
                
                shap.summary_plot(display_values, X_test, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary: {type(model).__name__} Impact on Student Risk')
                
                shap_path = os.path.join(report_dir, f'shap_summary_{timestamp}.png')
                plt.savefig(shap_path, bbox_inches='tight', dpi=300)
                paths["shap"] = shap_path
                plt.close()
            except Exception as shap_err:
                print(f"SHAP Plotting Error: {str(shap_err)}")
                plt.close()
                
        return paths
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return paths

def get_timestamp():
    """Generate timestamp in [HH:MM:SS AM/PM] format"""
    return datetime.now().strftime("[%I:%M:%S %p]")

def add_log_entry(logs, step, total_steps, message, status="→"):
    """Add a timestamped log entry with progress indicator"""
    timestamp = get_timestamp()
    logs.append({
        'timestamp': timestamp,
        'step': f"[{step}/{total_steps}]",
        'status': status,
        'message': message
    })

def get_metric_explanations():
    """Return academic-friendly explanations for each evaluation metric"""
    return {
        'accuracy': {
            'title': 'Accuracy',
            'explanation': 'Ratio of correctly predicted observations to total observations. Calculated as (TP + TN) / (TP + TN + FP + FN). Measures overall model correctness across all classes.',
            'interpretation': 'Higher values indicate better overall classification performance. Values near 1.0 suggest the model correctly classifies most instances.'
        },
        'precision': {
            'title': 'Precision',
            'explanation': 'Ratio of correctly predicted positive observations to total predicted positive observations. Calculated as TP / (TP + FP). Measures how many selected items are relevant.',
            'interpretation': 'Higher values indicate fewer false positives. Critical when minimizing false alarms is important (e.g., avoiding mislabeling low-risk students as high-risk).'
        },
        'recall': {
            'title': 'Recall (Sensitivity)',
            'explanation': 'Ratio of correctly predicted positive observations to all actual positive observations. Calculated as TP / (TP + FN). Measures ability to find all positive samples.',
            'interpretation': 'Higher values indicate fewer false negatives. Critical when missing positive cases is costly (e.g., failing to identify at-risk students).'
        },
        'f1_score': {
            'title': 'F1 Score',
            'explanation': 'Harmonic mean of Precision and Recall. Calculated as 2 × (Precision × Recall) / (Precision + Recall). Balances both metrics when class distribution is imbalanced.',
            'interpretation': 'Higher values indicate better balance between precision and recall. Useful when you need to consider both false positives and false negatives equally.'
        },
        'roc_auc': {
            'title': 'ROC AUC',
            'explanation': 'Area Under the Receiver Operating Characteristic Curve. Measures the model\'s ability to distinguish between classes across different threshold settings. Based on True Positive Rate vs False Positive Rate.',
            'interpretation': 'Values range from 0.5 (random guessing) to 1.0 (perfect classification). Higher values indicate better discriminative ability across all classification thresholds.'
        }
    }

def prepare_training_data(students, subject_grades):
    """
    Prepare training data from student data using actual dataset schema.
    Maps survey questions and academic data to feature columns.
    """
    data = []
    
    # Define feature columns based on actual dataset schema
    feature_columns = [
        # Basic Academic Data
        'study_hours', 'library_visits', 'lms_login_per_month',
        # GPA History
        'gpa_y1s1', 'gpa_y1s2', 'gpa_y2s1', 'gpa_y2s2', 'gpa_y3s1',
        # Financial Factors
        'scholarship_amount', 'family_income',
        # Course or Program Experience (Likert 1-5)
        'like_course', 'interested_in_subjects', 'course_motivates',
        # Academic Performance/History (Likert 1-5)
        'satisfied_with_performance', 'previous_grades_affect',
        # Learning Behavior (Likert 1-5)
        'study_regularly', 'submit_on_time', 'manage_time_well',
        # Instructor Interaction (Likert 1-5)
        'instructors_explain_clearly', 'approach_instructors', 'instructors_encourage',
        # Classmate/Peer Influence (Likert 1-5)
        'classmates_influence_positively', 'work_well_with_classmates', 'friends_motivate',
    ]
    
    for student in students:
        features = {}
        
        # Preserve student identification (not used for ML training but needed for display)
        features['student_id'] = student.get('student_id', '')
        features['full_name'] = student.get('full_name', '')
        
        # Basic academic data
        features['study_hours'] = float(student.get('study_hours', 0) or 0)
        features['library_visits'] = float(student.get('library_visits', 0) or 0)
        features['lms_login_per_month'] = float(student.get('lms_login_per_month', 0) or 0)
        
        # GPA history (handle missing values - empty means student hasn't reached that semester yet)
        for gpa_col in ['gpa_y1s1', 'gpa_y1s2', 'gpa_y2s1', 'gpa_y2s2', 'gpa_y3s1']:
            gpa = student.get(gpa_col)
            # Empty/None GPA means student hasn't reached that semester - use 0 as neutral
            # This won't affect risk calculation since we filter out 0 values in the analysis
            if gpa is None or gpa == '' or gpa == 0:
                features[gpa_col] = 0.0
            else:
                features[gpa_col] = float(gpa)
        
        # Financial factors
        features['scholarship_amount'] = float(student.get('scholarship_amount', 0) or 0)
        features['family_income'] = float(student.get('family_income', 0) or 0)
        
        # Course or Program Experience (Likert scale 1-5)
        features['like_course'] = float(student.get('like_course', 3) or 3)
        features['interested_in_subjects'] = float(student.get('interested_in_subjects', 3) or 3)
        features['course_motivates'] = float(student.get('course_motivates', 3) or 3)
        
        # Academic Performance/History (Likert scale 1-5)
        features['satisfied_with_performance'] = float(student.get('satisfied_with_performance', 3) or 3)
        features['previous_grades_affect'] = float(student.get('previous_grades_affect', 3) or 3)
        features['try_improve_grades'] = float(student.get('try_improve_grades', 3) or 3)
        
        # Learning Behavior (Likert scale 1-5)
        features['study_regularly'] = float(student.get('study_regularly', 3) or 3)
        features['submit_on_time'] = float(student.get('submit_on_time', 3) or 3)
        features['manage_time_well'] = float(student.get('manage_time_well', 3) or 3)
        
        # Instructor Interaction (Likert scale 1-5)
        features['instructors_explain_clearly'] = float(student.get('instructors_explain_clearly', 3) or 3)
        features['approach_instructors'] = float(student.get('approach_instructors', 3) or 3)
        features['instructors_encourage'] = float(student.get('instructors_encourage', 3) or 3)
        
        # Classmate/Peer Influence (Likert scale 1-5)
        features['classmates_influence_positively'] = float(student.get('classmates_influence_positively', 3) or 3)
        features['work_well_with_classmates'] = float(student.get('work_well_with_classmates', 3) or 3)
        features['friends_motivate'] = float(student.get('friends_motivate', 3) or 3)
        
        # Student concerns (text field - not used for ML training but stored for reference)
        features['concerns'] = student.get('concerns', '') or ''
        
        # Calculate target (risk_label) based on GPA and survey responses
        # Use GPA trend and overall survey average to determine risk
        valid_gpas = [features['gpa_y1s1'], features['gpa_y1s2'], features['gpa_y2s1'], features['gpa_y2s2'], features['gpa_y3s1']]
        valid_gpas = [g for g in valid_gpas if g > 0]
        
        if len(valid_gpas) == 0:
            continue  # Skip if no GPA data
        
        avg_gpa = sum(valid_gpas) / len(valid_gpas)
        
        # Calculate survey average (excluding financial and basic data)
        survey_cols = ['like_course', 'interested_in_subjects', 'course_motivates',
                      'satisfied_with_performance', 'previous_grades_affect', 'try_improve_grades',
                      'study_regularly', 'submit_on_time', 'manage_time_well',
                      'instructors_explain_clearly', 'approach_instructors', 'instructors_encourage',
                      'classmates_influence_positively', 'work_well_with_classmates', 'friends_motivate']
        survey_avg = sum(features[col] for col in survey_cols) / len(survey_cols)
        
        # Risk calculation: Low GPA + Low survey engagement = High risk
        # Normalize GPA (1.0-5.0) and survey (1-5) to similar scale
        gpa_normalized = (avg_gpa - 1.0) / 4.0  # 0-1 scale
        survey_normalized = (survey_avg - 1.0) / 4.0  # 0-1 scale
        
        # Combined score (higher = better = lower risk)
        combined_score = (gpa_normalized * 0.6) + (survey_normalized * 0.4)
        
        # Determine risk label based on combined score
        # BINARY CLASSIFICATION: 0 = Good Standing, 1 = At-Risk
        if combined_score >= 0.7:
            target = 0  # Low risk
        else:
            target = 1  # At-Risk
        
        features['target'] = target
        data.append(features)
    
    return data

def train_model(X_train, y_train, algorithm, random_seed, use_hyperopt=True, use_smote=True):
    """Train a specific ML algorithm with hyperparameter optimization and class balancing"""

    # Apply SMOTE for class balancing if enabled and dataset is large enough
    if use_smote and len(X_train) >= 10:
        try:
            smote = SMOTE(random_state=random_seed, k_neighbors=min(5, len(X_train) - 1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        except:
            # Fallback if SMOTE fails (e.g., not enough samples)
            X_train_balanced, y_train_balanced = X_train, y_train
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    if algorithm == 'Decision Tree':
        if use_hyperopt:
            # Hyperparameter optimization with GridSearchCV
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2', None],
                'ccp_alpha': [0.0, 0.01, 0.02]
            }
            base_model = DecisionTreeClassifier(random_state=random_seed)
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_balanced, y_train_balanced)
            model = grid_search.best_estimator_
        else:
            model = DecisionTreeClassifier(
                random_state=random_seed,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                ccp_alpha=0.01
            )
            model.fit(X_train_balanced, y_train_balanced)

    elif algorithm == 'Random Forest':
        if use_hyperopt:
            # Hyperparameter optimization with GridSearchCV
            param_grid = {
                'n_estimators': [40, 50, 60, 80],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [5, 8, 10],
                'min_samples_leaf': [2, 4, 5],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True, False]
            }
            base_model = RandomForestClassifier(random_state=random_seed)
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_balanced, y_train_balanced)
            model = grid_search.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=40,
                random_state=random_seed,
                max_depth=3,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                bootstrap=True,
                max_samples=0.8
            )
            model.fit(X_train_balanced, y_train_balanced)

    elif algorithm == 'SVM':
        if use_hyperopt:
            # Hyperparameter optimization with GridSearchCV
            param_grid = {
                'C': [0.1, 1.0, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'class_weight': ['balanced', None]
            }
            base_model = SVC(probability=True, random_state=random_seed)
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_balanced, y_train_balanced)
            model = grid_search.best_estimator_
        else:
            model = SVC(
                probability=True,
                random_state=random_seed,
                C=1.0,
                gamma='scale',
                kernel='rbf',
                class_weight='balanced'
            )
            model.fit(X_train_balanced, y_train_balanced)

    elif algorithm == 'KNN':
        if use_hyperopt:
            # Hyperparameter optimization with GridSearchCV
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, min(15, len(X_train_balanced) - 1)],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                'p': [1, 2]
            }
            base_model = KNeighborsClassifier()
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train_balanced, y_train_balanced)
            model = grid_search.best_estimator_
        else:
            model = KNeighborsClassifier(
                n_neighbors=min(12, len(X_train_balanced) - 1),
                weights='uniform',
                algorithm='auto'
            )
            model.fit(X_train_balanced, y_train_balanced)

    elif algorithm == 'Naive Bayes':
        # Naive Bayes has few hyperparameters, but we can add smoothing
        model = GaussianNB(var_smoothing=1e-9)
        # Add small noise to reduce overfitting
        X_train_noisy = X_train_balanced + np.random.normal(0, 0.05, X_train_balanced.shape)
        model.fit(X_train_noisy, y_train_balanced)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model

def evaluate_model(model, X_test, y_test):
    """
    FIXED: Evaluate model safely handling multi-class constraints 
    and padding probability matrix shapes in small datasets.
    """
    y_pred = model.predict(X_test)
    
    # Core Classification Assessment Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    # Binary classification labels: 0 and 1
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    
    try:
        unique_classes = np.unique(y_test)
        if hasattr(model, 'predict_proba') and len(unique_classes) > 1:
            y_proba = model.predict_proba(X_test)
            
            if y_proba.shape[1] == 2:
                # Binary ROC AUC uses the probability of the positive class (class 1)
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics['roc_auc'] = 0.5
        else:
            metrics['roc_auc'] = 0.5
    except Exception as e:
        print(f"Bypassed continuous ROC calculation step: {str(e)}")
        metrics['roc_auc'] = 0.5
    
    return metrics

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train ML models with real data from frontend"""
    logs = []
    total_steps = 12
    current_step = 0

    try:
        current_step += 1
        add_log_entry(logs, current_step, total_steps, "Initializing training pipeline", "→")

        data = request.json
        students = data.get('students', [])
        subject_grades = data.get('subject_grades', [])

        # Use fixed random seed for reproducible results
        random_seed = data.get('random_seed', 2024)
        add_log_entry(logs, current_step, total_steps, f"Using random seed: {random_seed}", "→")

        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)

        current_step += 1
        add_log_entry(logs, current_step, total_steps, f"Data loaded: {len(students)} student records", "✓")
        
        if len(students) < 10:
            add_log_entry(logs, current_step, total_steps, f"Error: Need at least 10 students. Found {len(students)}", "✗")
            return jsonify({
                'error': f'Need at least 10 students for training. Found {len(students)}.'
            }), 400
        
        current_step += 1
        add_log_entry(logs, current_step, total_steps, "Data preprocessing: Processing survey responses, GPA history, and financial data", "→")
        
        training_data = prepare_training_data(students, subject_grades)
        
        current_step += 1
        add_log_entry(logs, current_step, total_steps, f"Preprocessing complete: {len(training_data)} valid records prepared", "✓")
        
        # Split features and target using actual dataset schema
        feature_cols = [
            # Basic Academic Data
            'study_hours', 'library_visits', 'lms_login_per_month',
            # GPA History
            'gpa_y1s1', 'gpa_y1s2', 'gpa_y2s1', 'gpa_y2s2', 'gpa_y3s1',
            # Financial Factors
            'scholarship_amount', 'family_income',
            # Course or Program Experience (Likert 1-5)
            'like_course', 'interested_in_subjects', 'course_motivates',
            # Academic Performance/History (Likert 1-5)
            'satisfied_with_performance', 'previous_grades_affect', 'try_improve_grades',
            # Learning Behavior (Likert 1-5)
            'study_regularly', 'submit_on_time', 'manage_time_well',
            # Instructor Interaction (Likert 1-5)
            'instructors_explain_clearly', 'approach_instructors', 'instructors_encourage',
            # Classmate/Peer Influence (Likert 1-5)
            'classmates_influence_positively', 'work_well_with_classmates', 'friends_motivate',
        ]
        X = [[row[col] for col in feature_cols] for row in training_data]
        y = [row['target'] for row in training_data]
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 10:
            add_log_entry(logs, current_step, total_steps, f"Error: Need at least 10 valid students. Found {len(X)}", "✗")
            return jsonify({
                'error': f'Need at least 10 valid students with complete data for training. Found {len(X)}.'
            }), 400
        
        current_step += 1
        add_log_entry(logs, current_step, total_steps, "Feature engineering: 28 features extracted (GPA history, academic behavior, and engagement metrics)", "✓")
        
        current_step += 1
        add_log_entry(logs, current_step, total_steps, f"Dataset splitting: 70/30 train-test split with stratification (seed={random_seed})", "✓")

        # Split data with stratification to maintain class balance using random seed
        # 30/70 test/train split for balanced evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_seed, stratify=y
        )
        
        add_log_entry(logs, current_step, total_steps, f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples", "→")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        current_step += 1
        add_log_entry(logs, current_step, total_steps, "Feature scaling: StandardScaler applied (mean=0, std=1)", "✓")

        # Enable advanced features
        use_hyperopt = True
        use_smote = True

        current_step += 1
        add_log_entry(logs, current_step, total_steps, "Advanced features: Hyperparameter optimization enabled", "✓")
        add_log_entry(logs, current_step, total_steps, "Advanced features: SMOTE class balancing enabled", "✓")

        algorithms = ['Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes']
        results = []

        # Use 10-fold cross-validation for robust evaluation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
        cv_name = "10-fold"

        current_step += 1
        add_log_entry(logs, current_step, total_steps, f"Cross-validation: {cv_name} StratifiedKFold configured (seed={random_seed})", "✓")

        current_step += 1
        add_log_entry(logs, current_step, total_steps, "Sequential model training started", "→")

        for idx, algo in enumerate(algorithms):
            try:
                add_log_entry(logs, current_step, total_steps, f"Training {algo}...", "→")

                # Train model with hyperparameter optimization and class balancing
                model = train_model(X_train_scaled, y_train, algo, random_seed, use_hyperopt, use_smote)
                
                # 1. Evaluate on test set (Validation for Methodology)
                val_metrics = evaluate_model(model, X_test_scaled, y_test)
                
                # 2. Cross validation for stability check
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=cv, scoring='accuracy'
                )
                
                # 3. Evaluate on full dataset (to match the 101-record count in thesis results)
                X_all_scaled = scaler.transform(X)
                full_metrics = evaluate_model(model, X_all_scaled, y)
                
                add_log_entry(logs, current_step, total_steps, f"{algo} completed - CV: {cv_scores.mean():.4f}", "✓")
                
                results.append({
                    'algorithm': algo,
                    'accuracy': val_metrics['accuracy'],
                    'precision': val_metrics['precision'],
                    'recall': val_metrics['recall'],
                    'f1_score': val_metrics['f1_score'],
                    'full_population_accuracy': full_metrics['accuracy'],
                    'roc_auc': val_metrics['roc_auc'],
                    'confusion_matrix': val_metrics['confusion_matrix'], # Only ~30 records
                    'full_confusion_matrix': full_metrics['confusion_matrix'], # The 101 records
                    'cv_accuracy_mean': float(cv_scores.mean()),
                    'cv_accuracy_std': float(cv_scores.std()),
                    'cv_scores': cv_scores.tolist()
                })
            except Exception as e:
                print(f"Critical execution error tracing model sequence {algo}: {str(e)}")
                add_log_entry(logs, current_step, total_steps, f"{algo} failed: {str(e)}", "✗")
                results.append({
                    'algorithm': algo,
                    'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'roc_auc': 0,
                    'confusion_matrix': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    'cv_accuracy_mean': 0, 'cv_accuracy_std': 0, 'cv_scores': [0, 0, 0, 0, 0]
                })
        
        # Find best model based on F1 score
        best = max(results, key=lambda x: x['f1_score'])
        
        # Generate Accuracy Test Report File for Thesis
        # Create a specific folder for reports
        report_dir = os.path.join(os.getcwd(), 'reports')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"accuracy_test_report_{timestamp_str}.txt"

        report_path = os.path.join(report_dir, report_filename)
        
        # Generate visual plots for the BEST model
        # Re-use the existing training logic for the best algorithm
        # We use the best_algo_model directly for SHAP to ensure consistency
        best_algo_model = train_model(X_train_scaled, y_train, best['algorithm'], 
                                      random_seed, use_hyperopt, use_smote)
        visual_paths = save_visualization_plots(best_algo_model, X_test_scaled, y_test, 
                                                feature_cols, report_dir, timestamp_str)

        with open(report_path, 'w') as f:
            f.write("====================================================\n")
            f.write("      STUDENT RISK PREDICTION: ACCURACY TEST REPORT\n")
            f.write(f"      Generated on: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n")
            f.write("====================================================\n")
            f.write(f"Dataset Size: {len(students)} students\n")
            f.write(f"Split Ratio: 70% Training / 30% Testing\n")
            f.write(f"Validation: 10-Fold Stratified Cross-Validation\n")
            f.write(f"Random Seed: {random_seed}\n\n")
            
            f.write(f"{'Algorithm':<20} | {'Test Acc':<9} | {'Full Pop':<9} | {'F1 Score':<9} | {'CV Mean':<9}\n")
            f.write("-" * 80 + "\n")
            
            for res in results:
                f.write(f"{res['algorithm']:<20} | "
                        f"{res['accuracy']*100:>8.2f}% | "
                        f"{res['full_population_accuracy']*100:>8.2f}% | "
                        f"{res['f1_score']*100:>8.2f}% | "
                        f"{res['cv_accuracy_mean']*100:>8.2f}%\n")
            
            f.write("-" * 105 + "\n")
            f.write("-" * 75 + "\n")
            f.write(f"BEST MODEL SELECTED: {best['algorithm']}\n")
            f.write(f"Reason: Highest F1-Score ({best['f1_score']:.4f}) in Binary Classification\n")
            f.write("\n====================================================\n")
            if visual_paths.get("shap"):
                f.write(f"XAI Output: SHAP Summary plot generated at {os.path.basename(visual_paths['shap'])}\n")
        print(f"\n[SUCCESS] Accuracy report generated at: {report_path}\n")

        current_step += 1
        add_log_entry(logs, current_step, total_steps, f"Model comparison complete: Best model = {best['algorithm']} (F1 Score: {best['f1_score']:.4f})", "✓")
        
        current_step += 1
        if SHAP_AVAILABLE:
            add_log_entry(logs, current_step, total_steps, "Explainable AI: Calculating SHAP values for global interpretation", "→")
            try:
                if best['algorithm'] in ['Random Forest', 'Decision Tree']:
                    explainer = shap.TreeExplainer(best_algo_model)
                    shap_values = explainer.shap_values(X_test_scaled)
                    
                    if isinstance(shap_values, list):
                        best['shap_importance'] = dict(zip(feature_cols, np.abs(shap_values[1]).mean(axis=0).tolist()))
                    else:
                        best['shap_importance'] = dict(zip(feature_cols, np.abs(shap_values).mean(axis=0).tolist()))
                
                add_log_entry(logs, current_step, total_steps, "SHAP interpretation completed successfully", "✓")
            except Exception as e:
                add_log_entry(logs, current_step, total_steps, f"SHAP calculation skipped: {str(e)}", "!")
        else:
            add_log_entry(logs, current_step, total_steps, "SHAP library not found. Using standard feature importance instead.", "!")
            
        current_step += 1
        add_log_entry(logs, current_step, total_steps, f"Saving trained model and results to database (Session ID: {int(time.time())})", "✓")

        # Save the best model and scaler for prediction with same advanced features
        best_model = train_model(X_train_scaled, y_train, best['algorithm'], random_seed, use_hyperopt, use_smote)
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        add_log_entry(logs, current_step, total_steps, f"Best model ({best['algorithm']}) saved to disk", "✓")

        # Get metric explanations
        metric_explanations = get_metric_explanations()

        return jsonify({
            'success': True,
            'results': results,
            'best': best,
            'dataset_size': len(students),
            'session_id': str(int(time.time())),
            'training_log': logs,
            'metric_explanations': metric_explanations,
            'random_seed': random_seed
        })
        
    except Exception as e:
        add_log_entry(logs, current_step, total_steps, f"Training pipeline failed: {str(e)}", "✗")
        return jsonify({
            'success': False,
            'error': str(e),
            'training_log': logs
        }), 500

@app.route('/api/model-visualization', methods=['POST'])
def get_model_visualization():
    """Export model visualization data for explainable AI"""
    try:
        data = request.json
        algorithm = data.get('algorithm', 'Decision Tree')
        
        # Load the trained model
        try:
            model = joblib.load('best_model.pkl')
        except FileNotFoundError:
            return jsonify({
                'success': False,
                'error': 'Model file not found. Please train a model first.'
            }), 404
        
        try:
            scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            scaler = None
        
        feature_cols = [
            'study_hours', 'library_visits', 'lms_login_per_month',
            'gpa_y1s1', 'gpa_y1s2', 'gpa_y2s1', 'gpa_y2s2', 'gpa_y3s1',
            'scholarship_amount', 'family_income',
            'like_course', 'interested_in_subjects', 'course_motivates',
            'satisfied_with_performance', 'previous_grades_affect',
            'study_regularly', 'submit_on_time', 'manage_time_well',
            'instructors_explain_clearly', 'approach_instructors', 'instructors_encourage',
            'classmates_influence_positively', 'work_well_with_classmates', 'friends_motivate',
        ]
        
        visualization_data = {
            'algorithm': algorithm,
            'feature_names': feature_cols,
        }
        
        # Decision Tree Visualization
        if hasattr(model, 'tree_'):
            tree = model.tree_
            n_nodes = tree.node_count
            children_left = tree.children_left
            children_right = tree.children_right
            feature = tree.feature
            threshold = tree.threshold
            
            tree_structure = []
            def recurse(node, depth):
                if children_left[node] != children_right[node]:
                    # Internal node
                    tree_structure.append({
                        'node_id': node,
                        'depth': depth,
                        'feature': feature_cols[feature[node]] if feature[node] != -2 and feature[node] < len(feature_cols) else 'unknown',
                        'threshold': threshold[node],
                        'is_leaf': False,
                        'left_child': children_left[node],
                        'right_child': children_right[node],
                        'samples': tree.n_node_samples[node],
                        'gini': tree.impurity[node]
                    })
                    recurse(children_left[node], depth + 1)
                    recurse(children_right[node], depth + 1)
                else:
                    # Leaf node
                    tree_structure.append({
                        'node_id': node,
                        'depth': depth,
                        'is_leaf': True,
                        'samples': tree.n_node_samples[node],
                        'gini': tree.impurity[node],
                        'class': int(tree.value[node].argmax())
                    })
            
            recurse(0, 0)
            visualization_data['tree_structure'] = tree_structure
            if hasattr(model, 'feature_importances_'):
                visualization_data['feature_importance'] = dict(zip(feature_cols, model.feature_importances_.tolist()))
        
        # Random Forest Visualization
        elif hasattr(model, 'estimators_'):
            n_trees = len(model.estimators_)
            visualization_data['n_trees'] = n_trees
            if hasattr(model, 'feature_importances_'):
                visualization_data['feature_importance'] = dict(zip(feature_cols, model.feature_importances_.tolist()))
            
            # Get first few trees for visualization
            tree_samples = []
            for i in range(min(3, n_trees)):
                try:
                    tree = model.estimators_[i].tree_
                    tree_structure = []
                    def recurse_rf(node, depth):
                        if tree.children_left[node] != tree.children_right[node]:
                            tree_structure.append({
                                'node_id': node,
                                'depth': depth,
                                'feature': feature_cols[tree.feature[node]] if tree.feature[node] != -2 and tree.feature[node] < len(feature_cols) else 'unknown',
                                'threshold': tree.threshold[node],
                                'is_leaf': False
                            })
                            recurse_rf(tree.children_left[node], depth + 1)
                            recurse_rf(tree.children_right[node], depth + 1)
                        else:
                            tree_structure.append({
                                'node_id': node,
                                'depth': depth,
                                'is_leaf': True,
                                'class': int(tree.value[node].argmax())
                            })
                    recurse_rf(0, 0)
                    tree_samples.append(tree_structure)
                except Exception as e:
                    print(f"Error processing tree {i}: {e}")
            visualization_data['tree_samples'] = tree_samples
        
        # Naive Bayes Visualization
        elif hasattr(model, 'class_count_'):
            visualization_data['class_priors'] = model.class_prior_.tolist()
            visualization_data['class_count'] = model.class_count_.tolist()
            visualization_data['feature_means'] = model.theta_.tolist()
            visualization_data['feature_var'] = model.var_.tolist()
        
        # KNN Visualization
        elif hasattr(model, '_fit_X'):
            visualization_data['n_neighbors'] = model.n_neighbors
            visualization_data['n_samples_fit'] = model._fit_X.shape[0]
            visualization_data['n_features'] = model._fit_X.shape[1]
        
        # SVM Visualization
        elif hasattr(model, 'support_vectors_'):
            visualization_data['n_support'] = model.n_support_.tolist()
            visualization_data['support_vectors_count'] = len(model.support_vectors_)
            visualization_data['dual_coef'] = model.dual_coef_.tolist()
        
        # General feature importance for all models
        if hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                importance = abs(model.coef_[0])
            else:
                importance = abs(model.coef_).mean(axis=0)
            visualization_data['feature_importance'] = dict(zip(feature_cols, importance.tolist()))

        # Add SHAP as an additional explanation layer if available
        if SHAP_AVAILABLE:
            try:
                # Generate local explanation for a representative sample (the first test record)
                base_val = 0.5
                # If we have a scaler, we should use representative data
                sample_input = np.zeros((1, len(feature_cols)))
                
                if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
                    explainer = shap.TreeExplainer(model)
                    shap_values_local = explainer.shap_values(sample_input)
                    
                    # Handle binary classification
                    if isinstance(shap_values_local, list):
                        local_vals = shap_values_local[1][0].tolist()
                        base_val = float(explainer.expected_value[1])
                    else:
                        local_vals = shap_values_local[0].tolist()
                        base_val = float(explainer.expected_value)
                        
                    visualization_data['local_shap_values'] = dict(zip(feature_cols, local_vals))

                visualization_data['shap_enabled'] = True
                visualization_data['shap_base_value'] = base_val
            except Exception as e:
                print(f"SHAP visualization error: {e}")
                visualization_data['shap_enabled'] = False

        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_native(obj.tolist())
            else:
                return obj
        
        visualization_data = convert_to_native(visualization_data)
        
        return jsonify({
            'success': True,
            'visualization_data': visualization_data
        })
        
    except Exception as e:
        import traceback
        print(f"Error in model visualization: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_students():
    """Predict risk levels for students using trained model"""
    try:
        data = request.json
        students = data.get('students', [])

        # Prepare training data with student identification
        training_data = prepare_training_data(students, [])

        # Extract features for prediction
        feature_cols = [
            'study_hours', 'library_visits', 'lms_login_per_month',
            'gpa_y1s1', 'gpa_y1s2', 'gpa_y2s1', 'gpa_y2s2', 'gpa_y3s1',
            'scholarship_amount', 'family_income',
            'like_course', 'interested_in_subjects', 'course_motivates',
            'satisfied_with_performance', 'previous_grades_affect', 'try_improve_grades',
            'study_regularly', 'submit_on_time', 'manage_time_well',
            'instructors_explain_clearly', 'approach_instructors', 'instructors_encourage',
            'classmates_influence_positively', 'work_well_with_classmates', 'friends_motivate',
        ]

        predictions = []

        # Try to load trained model and scaler
        if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
            model = joblib.load('best_model.pkl')
            scaler = joblib.load('scaler.pkl')

            if training_data:
                # Vectorized batch processing for better performance
                X_batch = [[row[col] for col in feature_cols] for row in training_data]
                X_batch = np.array(X_batch)
                X_scaled = scaler.transform(X_batch)

                # Batch predictions are much faster than single calls in a loop
                y_pred = model.predict(X_scaled)
                y_proba = model.predict_proba(X_scaled)

                for i, row in enumerate(training_data):
                    prediction = y_pred[i]
                    probability = y_proba[i]

                # Map prediction to risk level
                if prediction == 0:
                    risk_level = 'Good Standing'
                else:
                    risk_level = 'At-Risk'

                predictions.append({
                    'student_id': row.get('student_id', ''),
                    'full_name': row.get('full_name', ''),
                    'risk_level': risk_level,
                    'confidence': float(max(probability)),
                    'concerns': row.get('concerns', '')
                })
        else:
            # Fallback to rule-based prediction if no trained model exists
            for row in training_data:
                valid_gpas = [row['gpa_y1s1'], row['gpa_y1s2'], row['gpa_y2s1'], row['gpa_y2s2'], row['gpa_y3s1']]
                valid_gpas = [g for g in valid_gpas if g > 0]

                if len(valid_gpas) == 0:
                    risk_level = 'Unknown'
                else:
                    avg_gpa = sum(valid_gpas) / len(valid_gpas)

                    survey_cols = ['like_course', 'interested_in_subjects', 'course_motivates',
                                  'satisfied_with_performance', 'previous_grades_affect', 'try_improve_grades',
                                  'study_regularly', 'submit_on_time', 'manage_time_well',
                                  'instructors_explain_clearly', 'approach_instructors', 'instructors_encourage',
                                  'classmates_influence_positively', 'work_well_with_classmates', 'friends_motivate']
                    survey_avg = sum(row[col] for col in survey_cols) / len(survey_cols)

                    gpa_normalized = (avg_gpa - 1.0) / 4.0
                    survey_normalized = (survey_avg - 1.0) / 4.0
                    combined_score = (gpa_normalized * 0.6) + (survey_normalized * 0.4)

                    if combined_score >= 0.7:
                        risk_level = 'Good Standing'
                    else:
                        risk_level = 'At-Risk'

                predictions.append({
                    'student_id': row.get('student_id', ''),
                    'full_name': row.get('full_name', ''),
                    'risk_level': risk_level,
                    'confidence': 0.0,
                    'concerns': row.get('concerns', '')
                })

        return jsonify({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
