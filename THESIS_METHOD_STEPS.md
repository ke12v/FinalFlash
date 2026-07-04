# Performance Test Methodology (Base sa System)

## 1. Overview sa Performance Test (Main Goal)
Ang Performance Test ani nga system kay aron i-evaluate pinaagi sa **structured, automated testing pipeline** ang **real performance** ug **stability** sa 4 ka part nga mao pud ang main scope:
1) **Model Training**
2) **Prediction Generation**
3) **Report Viewing**
4) **Log Viewing**


## 2. Training Phase (Performance Test Step-by-Step)
### Step 2.1: Pag-signal sa Training Request
- **Unsa ni:** Kini ang unang stage sa performance test.
- **Unsaon:** Ang frontend mo-send ug training request sa backend pinaagi sa API:
  - **`POST /api/train`**.
- **Unsa ang sulod sa request body:**
  - `students`: listahan sa student records (survey + GPA + financial).
  - `subject_grades`: optional/available sa system (ang training pipeline mo-focus sa available fields).
  - `random_seed`: para consistent ug reproducible ang results (default: 2024).


### Step 2.2: Reproducibility (Random Seeds)
- Gibutang ang fixed seeds:
  - `np.random.seed(random_seed)`
  - `random.seed(random_seed)`

### Step 2.3: Data Validation
- Gibalik sa system ang error kung **dili pa moabot ug minimum threshold** (diha sa code: kinahanglan **>= 10 students**).

### Step 2.4: Data Preprocessing ug Feature Preparation
- Ang system naggamit og function **`prepare_training_data(students, subject_grades)`**.
- Giprepare ang **feature columns** gikan sa student inputs, ilabi na:
  - Academic behavioral features (study hours, library visits, LMS login)
  - GPA history per semester (gpa_y1s1, y1s2, y2s1, y2s2, y3s1)
  - Financial features (scholarship_amount, family_income)
  - Survey Likert-scale features (study habits, instructor interaction, peer influence, etc.)
- Target label (**risk class**) gihimo pinaagi sa risk calculation:
  - Normalized GPA (0-1) ug survey average (0-1)
  - Combined score = `0.6*GPA_normalized + 0.4*survey_normalized`
  - Binary Label mapping:
    - `combined_score >= 0.5` → **0 (Good Standing)**
    - `combined_score < 0.5` → **1 (At-Risk)**

### Step 2.5: Feature Matrix and Target Vector
 - `X` gihimo as array sa features
 - `y` gihimo as target array (0/1)

### Step 2.6: Dataset Splitting (Train/Test)
- Gibuhat ang **train-test split**:
  - **70% Training**
  - **30% Testing**
- Gibutang ang stratification:
  - `train_test_split(... stratify=y, test_size=0.3, random_state=random_seed)`

### Step 2.7: Feature Scaling
- Gibuhat ang scaling gamit ang **`StandardScaler`**:
  - Fit sa training (`scaler.fit_transform(X_train)`)
  - Transform sa testing (`scaler.transform(X_test)`)

### Step 2.8: Advanced Training Techniques
#### 2.8.1 Hyperparameter Optimization
- Para sa mga algorithm nga naay hyperparameter search, ang system naggamit og **GridSearchCV** (cv=5) ug scoring: **`f1_weighted`**.

#### 2.8.2 Class Balancing (SMOTE)
- Kung enabled (`use_smote=True`) ug dataset size kay igo, ang system nag-apply ug **SMOTE** sa training set para ma-balanse ang classes.
- Kung mapakyas ang SMOTE (e.g., insufficient samples), naay fallback nga balik sa original training set.

## 3. Model Training Across Algorithms (Performance Test Step)
### Step 3.1: Algorithms nga Gi-evaluate
Ang system nagtrain ug multiple algorithms:
- Decision Tree
- Random Forest
- SVM
- KNN
- Naive Bayes

### Step 3.2: Cross-Validation Setup
- Naggamit ang system og **StratifiedKFold**:
  - `n_splits=10`
  - `shuffle=True`
  - `random_state=random_seed`

### Step 3.3: Model Training ug Test Evaluation per Algorithm
Para matag algorithm:
1) Train gamit ang `train_model(...)`
2) Evaluate sa **test set** gamit ang `evaluate_model(...)`
3) Compute cross-validation accuracy gamit ang:
   - `cross_val_score(... scoring='accuracy')`

## 4. Performance Metrics (Performance Test Metrics)
Sa `evaluate_model(...)`, ang system nag-compute ug:

1) **Accuracy**
- Ratio sa tama nga predictions over total instances.

2) **Precision (weighted)**
- Measures kaayo pagkamapili sa model sa positive class (weighted across classes).

3) **Recall (weighted)**
- Measures kaayo pag-capture sa model sa actual class instances.

4) **F1 Score (weighted)**
- Harmonic mean sa precision ug recall; tungod kay weighted, mas maayo kung imbalanced classes.

5) **ROC AUC (weighted, multi-class OVR)**
- Gihimo pinaagi sa probability output (`predict_proba`) kung available.
- Naghatag ug measure kung unsa ka maayo ang model sa pag-discriminate sa classes.

6) **Confusion Matrix (3x3)**
- `labels=[0,1,2]` para sure nga compatible sa visualization/front-end.
- Mapping:
  - 0: Good Standing
  - 1: Moderate Risk
  - 2: At-Risk

## 5. Model Selection Criteria (Performance Test Step)
### Step 5.1: Best Model Definition
- Ang best model gipili base sa **highest weighted F1 score**:
  - `best = max(results, key=lambda x: x['f1_score'])`

## 6. Report Generation ug Visualization (Performance Test Step)
### Step 6.1: Report File Creation
- Ang system mo-generate ug text report sa `reports/` folder:
  - `accuracy_test_report_<timestamp>.txt`

### Step 6.2: Confusion Matrix Plot
- Para sa **best model**, gihimo ang confusion matrix heatmap.
- Output: PNG sa `reports/`.

### Step 6.3: Feature Importance Plot
- Para sa Random Forest/feature importance compatible models:
  - Top features (default top 15) ginabutang sa bar plot.
- Output: PNG sa `reports/`.

## 7. Prediction Generation Phase (Performance Test Step)
### Step 7.1: Prediction Request
- Ang prediction request moadto sa:
  - **`POST /api/predict`**.

### Step 7.2: Model + Scaler Loading
- Kung naa ang saved artifacts:
  - `best_model.pkl`
  - `scaler.pkl`
- I-scale ang features gamit ang saved scaler.
- Get predicted class ug probability:
  - `model.predict(...)`
  - `model.predict_proba(...)`

### Step 7.3: Risk Level Mapping
- Predicted label to risk string:
  - 0 → Good Standing
  - 1 → At-Risk

### Step 7.4: Prediction Output
- Ang system mo-respond og listahan sa students uban ang:
  - `student_id`
  - `full_name`
  - `risk_level`
  - `confidence`
  - `concerns`

## 8. Log Viewing Phase (Performance Test Step)
### Step 8.1: Training Logs in Response
- Ang training endpoint nagbalik ug `training_log`:
  - array nga nagpakita ug step indicator (`[step/total]`), timestamp, ug message.

### Step 8.2: Purpose sa Logs
- Para masklaro sa user/researcher ang:
  - pipeline stages
  - dataset preprocessing progress
  - CV setup
  - algorithm-specific completion status

## 9. Summary of Performance Test Workflow
**Performance Test Flow =**
1) **Model Training** (`/api/train`)
2) **Prediction Generation** (`/api/predict`) using saved `best_model.pkl` + `scaler.pkl`
3) **Report Viewing** (reports folder + generated PNGs)
4) **Log Viewing** (training_log returned by `/api/train`)

---
*Note: Kini nga methodology textual ra ug naka-base sa existing system behavior. Ang code/config wala gibag-o.*
