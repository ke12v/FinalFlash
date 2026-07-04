import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
import matplotlib

# Use non-interactive backend for file generation
matplotlib.use('Agg')

def generate_shap_png():
    # 1. Path Configuration
    base_path = r'c:\Users\KEVIN\Downloads\FinalFlash'
    model_path = os.path.join(base_path, 'best_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    output_path = os.path.join(base_path, 'reports', 'shap_summary_report.png')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: best_model.pkl or scaler.pkl not found. Please train your model via the API first.")
        return

    # 2. Load trained components
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3. Define feature columns (Must match the 25 features used in app.py)
    feature_names = [
        'study_hours', 'library_visits', 'lms_login_per_month',
        'gpa_y1s1', 'gpa_y1s2', 'gpa_y2s1', 'gpa_y2s2', 'gpa_y3s1',
        'scholarship_amount', 'family_income',
        'like_course', 'interested_in_subjects', 'course_motivates',
        'satisfied_with_performance', 'previous_grades_affect', 'try_improve_grades',
        'study_regularly', 'submit_on_time', 'manage_time_well',
        'instructors_explain_clearly', 'approach_instructors', 'instructors_encourage',
        'classmates_influence_positively', 'work_well_with_classmates', 'friends_motivate',
    ]

    # 4. Generate sample data for the summary plot
    # In a production scenario, you would load your actual X_test here.
    # For this report, we generate a synthetic distribution to visualize feature impact.
    X_sample = np.random.uniform(1, 4, size=(100, len(feature_names)))
    X_scaled = scaler.transform(X_sample)

    # 5. Calculate SHAP values
    print("Calculating SHAP values (this may take a moment)...")
    try:
        # Select appropriate explainer based on model type
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            # Optimized for Decision Trees and Random Forests
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        else:
            # Generic explainer for SVM, KNN, Naive Bayes, etc.
            explainer = shap.Explainer(model.predict, X_scaled)
            shap_values = explainer(X_scaled).values

        # For binary classification, SHAP returns a list [class0, class1]
        # We visualize Class 1 (At-Risk) impact
        if isinstance(shap_values, list):
            display_values = shap_values[1]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3: # (samples, features, classes)
            display_values = shap_values[:, :, 1]
        else:
            display_values = shap_values

        # 6. Create Plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            display_values, 
            X_scaled, 
            feature_names=feature_names, 
            show=False
        )
        plt.title(f'SHAP Summary: Feature Impact on Student Risk', fontsize=16, pad=20)

        # 7. Save PNG
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Success! SHAP summary plot saved to: {output_path}")

    except Exception as e:
        print(f"Failed to generate SHAP plot: {str(e)}")

if __name__ == "__main__":
    generate_shap_png()
