# 23CSE301 Machine Learning Capstone

## ML-301

This repository contains the implementation and evaluation work for the
**23CSE301 Machine Learning Capstone — Academic Year 2026–27**.

The project follows an end-to-end Machine Learning workflow covering
data analysis, preprocessing, feature engineering, model training,
evaluation, hyperparameter tuning, and model comparison.

---

## Project Structure

```text
ML-301/
│
├── app/
│   └── Application files
│
├── data/
│   └── health_insurance.csv
│
├── models/
│   └── Saved model files
│
├── notebooks/
│   ├── classification/
│   │   ├── 00_feature_distributions.ipynb
│   │   ├── 01_logistic_regression.ipynb
│   │   ├── 02_knn_classifier.ipynb
│   │   ├── 03_gaussian_naive_bayes.ipynb
│   │   ├── 04_decision_tree_classifier.ipynb
│   │   ├── 05_svc.ipynb
│   │   └── 06_classification_comparison.ipynb
│   │
│   └── regression/
│       ├── 00_data_loading_eda.ipynb
│       ├── 00_preprocessing.ipynb
│       ├── 01_linear_regression.ipynb
│       ├── 02_ridge_regression.ipynb
│       ├── 03_lasso_regression.ipynb
│       ├── 04_elasticnet_regression.ipynb
│       └── ...
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
│       └── Regression model results
│
├── .gitignore
├── README.md
├── REVIEW1_WORK_SPLIT.md
└── requirements.txt
