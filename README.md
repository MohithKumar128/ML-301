# 23CSE301 Machine Learning Capstone

## ML-301

This repository contains the work for the **23CSE301 Machine Learning Capstone — Academic Year 2026–27**. It covers data analysis, preprocessing, model development, evaluation, and comparison across regression, classification, and clustering.

## Project Objectives

- Analyse datasets using exploratory data analysis (EDA).
- Clean and preprocess data.
- Train and compare machine learning algorithms.
- Evaluate models and perform hyperparameter tuning.
- Identify patterns and groups using clustering.
- Maintain reproducible experiments and documented results.

## Datasets

### 1. Regression — Health Insurance Dataset
Predicts the continuous target variable `claim` using health insurance features.

### 2. Classification — Diabetes Risk Prediction Dataset
Predicts diabetes risk categories. Refer to the classification notebook for the exact target labels and preprocessing.

### 3. Clustering — Bank Marketing Dataset
Uses `bank-additional-full.csv` to discover groups of records based on their features. The target column `y` is excluded when fitting clustering algorithms.

## Project Workflows

### Regression
Implements and compares Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN Regression.

Notebook: `notebooks/regression.ipynb`

### Classification
Evaluates 10 classification algorithms, including Logistic Regression, KNN, Naive Bayes, Decision Tree, SVC, Random Forest, AdaBoost, Gradient Boosting, Bagging, and MLP. Model selection and hyperparameter tuning are also included.

Notebook: `notebooks/classification.ipynb`

### Clustering
Implements K-Means and Agglomerative Clustering. Analysis includes the Elbow Method, dendrograms, Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index, PCA visualisations, and t-SNE visualisation.

Notebook: `notebooks/clustering.ipynb`

## Repository Structure

```text
ML-301/
├── app/
├── data/
├── models/
├── notebooks/
│   ├── classification.ipynb
│   ├── regression.ipynb
│   └── clustering.ipynb
├── results/
│   ├── classification_results/
│   ├── regression/
│   └── clustering/
├── README.md
├── requirements.txt
├── REVIEW1_WORK_SPLIT.md
└── .gitignore
```

## Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/MohithKumar128/ML-301.git
cd ML-301
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

## How to Run

1. Ensure the required datasets are available at the paths expected by the notebooks.
2. Open the relevant notebook from `notebooks/`.
3. Run cells from top to bottom.
4. Review the generated results and visualisations in `results/`.

## Reproducibility

- Use fixed random seeds where applicable.
- Fit preprocessing transformations on training data only for supervised workflows.
- Exclude target variables when fitting clustering algorithms.
- Verify notebook outputs and saved results before submission.

## Conclusion

This project demonstrates an end-to-end machine learning workflow across regression, classification, and unsupervised clustering, with a focus on model comparison, evaluation, and reproducibility.