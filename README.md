````markdown
# 23CSE301 Machine Learning Capstone

## ML-301

This repository contains the implementation and evaluation work for the **23CSE301 Machine Learning Capstone — Academic Year 2026–27**.

The project follows an end-to-end Machine Learning workflow covering data analysis, preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and model comparison.

---

## Review 1

The main aim of **Review 1** is to establish the foundation of the Machine Learning project through dataset analysis, preprocessing, feature engineering, and initial model development and evaluation.

Review 1 focuses on:

- Dataset understanding and Exploratory Data Analysis
- Data cleaning and preprocessing
- Missing-value and outlier handling
- Feature engineering
- Regression model development and evaluation
- Classification Part-A model development and evaluation
- Initial model comparison
- Hyperparameter tuning for selected regression models

---

## Datasets

### Regression Dataset

**Dataset:** Health Insurance Data Set

The regression track uses health insurance data for predicting a continuous target variable called claim.

### Classification Dataset

**Dataset:** Diabetes Risk Prediction Dataset

The classification track predicts the diabetes risk category of a patient diabetes as low  ,middle and high.

---

## Project Structure

```text
ML-301/
│
├── .venv/
│
├── app/
│
├── data/
│   ├── .gitkeep
│   ├── diabetes_risk_prediction_dataset.csv
│   └── health_insurance.csv
│
├── models/
│
├── notebooks/
│   ├── .ipynb_checkpoints/
│   ├── regression/
│   ├── .gitkeep
│   ├── classification.ipynb
│   └── regression.ipynb
│
├── results/
│   ├── classification_results/
│   │   ├── 01_logistic_regression.csv
│   │   ├── 02_knn_classifier.csv
│   │   ├── 03_gaussian_naive_bayes.csv
│   │   ├── 04_decision_tree_classifier.csv
│   │   └── 05_svc.csv
│   │
│   └── regression/
│       ├── 00_preprocessing_summary.csv
│       ├── 01_linear_regression.csv
│       ├── 02_ridge_regression.csv
│       ├── 03_lasso_regression.csv
│       ├── 04_elasticnet_regression.csv
│       ├── 05_polynomial_regression.csv
│       ├── 06_decision_tree_regressor.csv
│       ├── 07_random_forest_regressor.csv
│       ├── 08_gradient_boosting_regressor.csv
│       ├── 09_svr.csv
│       └── 10_knn_regressor.csv
│
├── .gitignore
├── README.md
├── requirements.txt
└── REVIEW1_WORK_SPLIT.md
````