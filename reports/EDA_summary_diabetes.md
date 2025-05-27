# 🧠 Diabetes Prediction – Exploratory Data Analysis

Dataset from Kaggle: [Type 2 Diabetes Dataset (Raw and Cleaned Versions)](https://www.kaggle.com/datasets/shahnawaj9/diabetes-database?resource=download)

## 📌 Problem Statement

This project aims to predict the presence of diabetes using clinical variables collected from female Pima Indian patients. Early detection can improve healthcare outcomes by guiding early intervention and monitoring.

The dataset contains 768 records and includes features like glucose level, insulin, BMI, and number of pregnancies. The `Outcome` column indicates whether the patient has diabetes (1) or not (0).

---

## 📊 Dataset Overview

- **Rows**: 768  
- **Columns**: 9 (8 features + 1 target)
- **Target**: `Outcome` (binary: 1 = diabetes, 0 = no diabetes)

### 🔢 Features:
| Name                        | Description                               |
|-----------------------------|-------------------------------------------|
| `Pregnancies`               | Number of pregnancies                     |
| `Glucose`                   | Plasma glucose concentration              |
| `BloodPressure`             | Diastolic blood pressure (mm Hg)          |
| `SkinThickness`             | Triceps skinfold thickness (mm)           |
| `Insulin`                   | 2-Hour serum insulin (mu U/ml)            |
| `BMI`                       | Body Mass Index                           |
| `DiabetesPedigreeFunction`  | Genetic predisposition to diabetes        |
| `Age`                       | Age in years                              |

---

## 🔍 Initial Observations

- No null values in `.info()`, but multiple features have **invalid 0 values** (e.g., Glucose, Insulin, BMI) which are biologically implausible.
- These 0s will be treated as missing in preprocessing using `np.nan`.
- Outliers exist in:
  - `Insulin`: some values exceed 800
  - `BMI`: some values above 60
  - `Glucose`: extends to 199

---

## 📈 Visual Insights

### ▶️ Glucose vs Diabetes
![Glucose vs Diabetes](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/diabetes/glucose_vs_diabetes.png)

- Diabetic patients (Outcome = 1) tend to have significantly higher glucose levels.

### ▶️ Pairplot of Selected Features
![Pairplot](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/diabetes/pairplot.png)

- `Glucose` and `BMI` show some class separation
- `Age`, `Insulin`, and `DiabetesPedigreeFunction` offer weaker individual separation but may help in combination

---

## 🧮 Target & Features

- **Target**: `Outcome`
- **Features to use**:
  - All except `Outcome`
  - Will need to treat 0s as missing in: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
 
---

## 🔄 PCA Analysis

### Diabetes Dataset

- PCA shows that the **first 4 principal components explain over 70%** of the variance.
- **All 8 components are needed to capture 100%**, but most of the useful variance is concentrated in the first few.
- This suggests **redundancy** among features (i.e., some are correlated and could be combined).
- If dimensionality reduction is desired, retaining **4–5 components** might preserve most of the signal while reducing complexity.
- However, for interpretability, it may be preferable to keep the original variables.

![PCA](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/diabetes/pca_diabetes.png)

![PC1_PC2](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/diabetes/pc1_pc2.png)

### Decision
- PCA will **not** be used to reduce dimensions for the baseline model, but this analysis helps confirm which features are more informative.