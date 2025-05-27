# ❤️ Heart Disease Prediction – Exploratory Data Analysis

Dataset from Kaggle: [Common Heart Disease Data (4 Hospitals)](https://www.kaggle.com/datasets/denysskyrda/common-heart-disease-data-4-hospitals/data)

## 📌 Problem Statement

This analysis aims to predict whether a patient has heart disease based on various clinical measurements. Accurate prediction models can help prioritize high-risk individuals for further screening and preventive care.

We use a dataset of 920 patients with features such as cholesterol, max heart rate, and exercise-induced angina, along with a binary `target` variable (1 = disease, 0 = no disease).

---

## 📊 Dataset Overview

- **Rows**: 920  
- **Columns**: 15 (13 features + `target` + `source`)
- **Target**: `target` (binary: 1 = heart disease, 0 = no heart disease)

### 🔢 Features:
| Name       | Description |
|------------|-------------|
| `age`      | Age in years |
| `sex`      | 0 = female, 1 = male |
| `cp`       | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol`     | Serum cholesterol (mg/dL) |
| `fbs`      | Fasting blood sugar > 120 mg/dL |
| `restecg`  | Resting ECG result |
| `thalach`  | Max heart rate achieved |
| `exang`    | Exercise-induced angina |
| `oldpeak`  | ST depression induced by exercise |
| `slope`    | Slope of the ST segment |
| `ca`       | Number of major vessels (0–3) |
| `thal`     | Thalassemia status |

---

## 🔍 Initial Observations

- No missing values were detected (`df.info()` and `.isnull().sum()` show full completeness).
- `chol`, `trestbps`, and `thalach` have values down to 0, which are **biologically implausible** and should be treated as missing in preprocessing.
- Some columns like `chol` have strong outliers (max = 603 mg/dL).
- `thalach` and `oldpeak` distributions differ between patients with and without heart disease.

---

## 📈 Visual Insights

### ▶️ Age vs Heart Disease
![Age vs Heart Disease](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/age_vs_heart.png)

- People with heart disease tend to be slightly older.

### ▶️ Cholesterol vs Heart Disease
![Cholesterol](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/cholesterol_vs_heart.png)

- Wide spread of cholesterol levels; many outliers above 400.

### ▶️ Max Heart Rate vs Heart Disease
![Max HR](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/max_hr_vs_heart.png)

- Patients without heart disease often have higher max heart rates.

### ▶️ Resting Blood Pressure
![Trestbps](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/trestbps_vs_heart.png)

- Slightly higher in patients with heart disease.

### ▶️ Pairplot Summary
![Pairplot](/Users/manuelrodriguezsutil/Developer/health-risk/visuals/heart_disease/pairplot.png)

- `oldpeak` and `thalach` show some separation between classes.
- `chol` and `age` alone are less discriminative.

---

## 🧮 Target & Features

- **Target**: `target`
- **Features to keep**:
  - All 13 patient features (after handling invalid 0s in continuous vars)
  - Drop `source` (metadata only)
 
---

### 🔄 PCA Analysis – Heart Disease Dataset

We applied PCA to the five continuous features:
- `age`, `trestbps`, `chol`, `thalach`, `oldpeak`

#### 🔍 Key Insights:

- The first **3 principal components explain ~72%** of the variance.
- The first **4 components cover ~89%**, and all 5 reach 100%.
- This suggests a moderate amount of redundancy or correlation among the continuous variables.

#### 📌 Decision:

- PCA helps us understand variance structure, but we will **not use it for dimensionality reduction** in the baseline model to maintain interpretability.
- PCA loadings (not shown here) indicated that:
  - `oldpeak` and `thalach` contributed strongly to PC1
  - `chol` and `trestbps` were more prominent in PC2