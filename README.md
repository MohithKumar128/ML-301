# ML-301: Machine Learning Capstone

This project explores machine learning techniques for regression, classification, and clustering. It includes data preprocessing, model training, performance evaluation, model comparison, and visualisation of results.

## Problem Statement

The aim of this project is to apply and compare machine learning algorithms to solve three tasks:

- **Regression:** Predict health insurance claim amounts from available features.
- **Classification:** Predict diabetes risk categories using patient-related features.
- **Clustering:** Discover groups of similar records in the Bank Marketing dataset without using the target variable to form clusters.

The project compares different algorithms using suitable evaluation metrics to identify models that perform well for each task.

## Datasets

| Task | Dataset | Objective |
|---|---|---|
| Regression | Health Insurance dataset | Predict insurance claim amounts |
| Classification | Diabetes Risk Prediction dataset | Predict diabetes risk categories |
| Clustering | Bank Marketing dataset (`bank-additional-full.csv`) | Identify groups of similar records |

## Algorithms Used

### 1. Regression

The following algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbours (KNN) Regressor

**Best-performing model: Random Forest Regressor**

Saved evaluation results:

- R² score: 0.9636
- RMSE: 2332.40
- MAE: 543.85
- Cross-validation R²: 0.9627

The Random Forest Regressor achieved the highest R² score among the listed regression models in the saved results, with lower prediction error than the other evaluated models.

### 2. Classification

The following algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbours (KNN)
- Gaussian Naive Bayes
- Decision Tree Classifier
- Support Vector Classifier (SVC)
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- Bagging Classifier
- Multi-Layer Perceptron (MLP)

**Best-performing model: Tuned AdaBoost Classifier**

Saved evaluation results:

- Accuracy: 96.10%
- Weighted F1-score: 95.71%
- Weighted ROC-AUC: 98.77%
- Best hyperparameters: `n_estimators = 200`, `learning_rate = 1.0`

AdaBoost was selected based on the comparison of the ten classification algorithms. Hyperparameter tuning improved its performance compared with the baseline model. However, the results also show that the minority Low-risk class remains difficult to classify.

### 3. Clustering

The following algorithms were evaluated:

- K-Means Clustering
- Agglomerative Clustering

The analysis includes:

- Elbow Method for selecting the number of clusters
- Dendrogram for hierarchical clustering
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index
- PCA-based two-dimensional visualisations
- t-SNE visualisation

**Current leading model: Agglomerative Clustering (Ward linkage)**

Saved evaluation results:

- Silhouette Score: 0.2496
- Davies–Bouldin Index: 1.4964

Agglomerative Clustering has a higher Silhouette Score and lower Davies–Bouldin Index than K-Means in the recorded comparison. These metrics suggest better-defined clusters on those two measures.

## Project Structure

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

## Environment Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Git
- Jupyter Notebook

### 1. Clone the repository

```bash
git clone https://github.com/MohithKumar128/ML-301.git
cd ML-301
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How to Run the Project

### Run the notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the relevant notebook from the `notebooks/` directory:

- `regression.ipynb`
- `classification.ipynb`
- `clustering.ipynb`

Ensure the required datasets are available in the expected locations before running the notebook cells. Run the cells in order and review the generated metrics, tables, and visualisations.

### Run the Streamlit application

If you want to launch the project application, run:

```bash
streamlit run app/app.py
```

## Evaluation Metrics

| Task | Metrics |
|---|---|
| Regression | R² score, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) |
| Classification | Accuracy, Weighted Precision, Weighted Recall, Weighted F1-score, ROC-AUC |
| Clustering | Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index |

For regression, higher R² and lower RMSE and MAE generally indicate better performance. For classification, multiple metrics are considered because the dataset contains imbalanced classes. For clustering, the metrics help assess cluster separation, compactness, and structure.

## Reproducibility

- Use fixed random seeds where appropriate.
- Fit preprocessing transformations on training data only for supervised learning tasks.
- Exclude the target variable when fitting clustering algorithms.
- Keep dependencies documented in `requirements.txt`.
- Review saved results and notebook outputs when comparing model performance.

## Conclusion

This project compares machine learning algorithms across regression, classification, and clustering. The saved results identify Random Forest Regressor as the leading regression model, tuned AdaBoost as the leading classification model, and Agglomerative Clustering (Ward linkage) as the current leader on the recorded Silhouette and Davies–Bouldin metrics.
