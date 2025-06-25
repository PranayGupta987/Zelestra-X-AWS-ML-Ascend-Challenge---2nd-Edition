# 🔋 Predictive Solar Panel Optimization using Machine Learning  
**Zelestra X AWS ML Ascend Challenge – 2nd Edition (Rank #63/1016)**  
Team **Beta**: Pranay, Ayush, Animesh

---

## 🚀 Project Overview

This project tackles the critical issue of **predicting performance degradation and failures in solar panels** using real-time and historical sensor data. It was developed as part of the **Zelestra X AWS ML Ascend Challenge**, where our team ranked **Top 6% (63/1016 teams)**.

> 🔍 Objective: Enable predictive maintenance and maximize solar energy output using a robust ML pipeline.

---

## 📁 Dataset Description

- `train.csv`: 20,000 rows × 17 columns (including target)
- `test.csv`: 12,000 rows × 16 columns
- `sample_submission.csv`: Template for final predictions

### 📌 Key Features
| Column Name             | Description |
|-------------------------|-------------|
| `temperature`           | Ambient air temperature (°C) |
| `irradiance`            | Solar energy per unit area (W/m²) |
| `humidity`, `pressure`  | Atmospheric conditions |
| `panel_age`, `maintenance_count` | Degradation history |
| `soiling_ratio`         | Dust/debris impact (0–1) |
| `voltage`, `current`    | Electrical output |
| `cloud_coverage`, `wind_speed` | Environmental factors |
| `string_id`, `error_code`, `installation_type` | Categorical identifiers |
| `efficiency`            | ⚠️ Target variable |

---

## 🧠 ML Pipeline Summary

### 🔧 Feature Engineering
- Manual domain-based features (e.g., `power_output`, `cooling_effect`, `expected_efficiency`)
- Binary indicators for conditions (e.g., `is_soiled`, `is_cloudy`)
- Polynomial interaction terms (5 features → 10 new features)

> **Expanded 17 original features to 60+ engineered features**

### 🔁 Preprocessing
- Robust scaling for numeric features
- One-hot encoding for categorical features
- Smart imputation (mean/mode)

### 🧪 Feature Selection
Multi-model importance blending:
- SHAP values from XGBoost
- Permutation importance
- Mutual Information
- LGBM and CatBoost feature importance
- SHAP interaction values
- RFE (Recursive Feature Elimination) with Ridge

### 📦 Modeling
- **Ensemble via `StackingRegressor`**  
  Base Models:
  - `XGBoost`
  - `LightGBM`
  - `CatBoost`
  Final Estimator:
  - `Ridge Regression (α optimized with Optuna)`

### 🧪 Hyperparameter Tuning
- Performed with `Optuna` over 20 trials
- Objective: minimize RMSE

---

## 📊 Evaluation

- Custom evaluation metric:  
  `Score = 100 × (1 - RMSE)`
- Achieved **final RMSE < 0.101**
- Final submission matched format: `12000 x 2` CSV

---

## 🛠 Tech Stack

- Python 🐍
- scikit-learn, XGBoost, LightGBM, CatBoost
- Optuna (hyperparameter optimization)
- SHAP (explainability)
- Pandas, NumPy, Matplotlib

---

## 🧠 Learnings & Highlights

- Built a modular ML pipeline that supports scalability and explainability.
- Blended multiple models and feature importance techniques for robust generalization.
- Applied real-time predictive analytics to a sustainability problem.
- Learned how to iterate quickly under a time-bound competitive setting.

---

## 👨‍💻 Authors

**Team Beta**  
- [Pranay Gupta](https://github.com/PranayGupta987)  
- [Ayush](https://github.com/AYUSH191004)  
- [Animesh](https://github.com/animesh8787)  


