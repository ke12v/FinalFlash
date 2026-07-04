# Per-Class Classification Report of the Best-Performing Model

| Class         | Precision | Recall | F1-Score | Support |
| ------------- | --------: | -----: | -------: | ------: |
| Good Standing |      0.94 |   1.00 |     0.97 |      15 |
| Mo|
| At-Risk       |      1.00 |   0.80 |     0.89 |       5 |
| **Accuracy**  |           |        | **0.94** |  **31** |

**1. Name of the best-performing model:** Random Forest (Optimized)  
**2. Dataset used:** `student_performance_dataset_v2.csv` (101 student records)  
**3. Train-test split used:** 70% Training / 30% Testing (Stratified)  
**4. SMOTE applied only to training data:** Yes. The system splits the data before passing the training set to the SMOTE balancer.  

**5. Python code used to generate this report:**
```python
from sklearn.metrics import classification_report
import pandas as pd
from pathlib import Path

# Logic extracted from algorithm_test_runner.py
target_names = ['Good Standing', 'Moderate Risk', 'At-Risk']
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_markdown("reports/per_class_classification_report.md")
```

**6. Interpretation:**  
The model exhibits high precision for the "At-Risk" class (1.00), meaning it never incorrectly flags a successful student as being at risk. While the recall for "At-Risk" (0.80) suggests a small number of students might be missed, the overall F1-score of 0.9355 demonstrates that the Random Forest algorithm is exceptionally robust at distinguishing between the three categories of academic performance.
